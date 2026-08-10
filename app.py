# app.py
import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import math
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config / Models (load once)
# ---------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast
    # NLI model
    nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    return embed_model, nli_tokenizer, nli_model

embed_model, nli_tokenizer, nli_model = load_models()

# Label order used by roberta-large-mnli: [contradiction, neutral, entailment]
NLI_LABELS = ["contradiction", "neutral", "entailment"]

# ---------------------------
# Helpers: PDF -> text
# ---------------------------
def extract_text_from_pdf_bytes(pdf_bytes):
    """Return a dict: {page_number: text} and also full text."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    full_text = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text().strip()
        pages[i+1] = text
        full_text.append(text)
    return pages, "\n\n".join(full_text)

# ---------------------------
# Helpers: chunking
# ---------------------------
def chunk_text(text, max_chars=800):
    """Naive chunking by sentences / fixed char size"""
    text = text.replace("\r", " ")
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += (" " + p) if current else p
        else:
            if current:
                chunks.append(current.strip())
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars].strip())
                current = ""
            else:
                current = p
    if current:
        chunks.append(current.strip())
    return chunks

# ---------------------------
# Build / Update vector index
# ---------------------------
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# ---------------------------
# NLI inference helper
# ---------------------------
@st.cache_resource
def run_nli_batch(evidence_list, claim, tokenizer=None, model=None, device=None):
    """Return list of dicts with probabilities for contradiction, neutral, entailment for each evidence."""
    if tokenizer is None or model is None:
        tokenizer = nli_tokenizer
        model = nli_model
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    model.eval()
    if device >= 0:
        model.to(torch.device("cuda"))
    results = []
    with torch.no_grad():
        for ev in evidence_list:
            premise = ev
            hypothesis = f"This sentence supports the claim: \"{claim}\""
            encoding = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            if device >= 0:
                encoding = {k: v.to("cuda") for k, v in encoding.items()}
            outputs = model(**encoding)
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            results.append({
                "contradiction": float(probs[0]),
                "neutral": float(probs[1]),
                "entailment": float(probs[2])
            })
    return results

# ---------------------------
# Streamlit UI / App logic
# ---------------------------
st.set_page_config(page_title="MythBuster AI - Research Paper Verifier", layout="wide")
st.title("🧠 MythBuster AI — Scientific Claim Verifier (using Research Papers)")

# Sidebar: upload and settings
st.sidebar.header("📚 Upload Research Papers")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s) of research papers", accept_multiple_files=True, type=["pdf"])

top_k = st.sidebar.slider("Number of evidence snippets to retrieve (k)", min_value=1, max_value=10, value=3)
chunk_size = st.sidebar.slider("Chunk size (chars)", min_value=200, max_value=1500, value=800)

if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "kb" not in st.session_state:
    st.session_state.kb = []

# Build index
if uploaded_files:
    all_chunks = []
    metadata = []
    for up in uploaded_files:
        pdf_bytes = up.read()
        pages, full_text = extract_text_from_pdf_bytes(pdf_bytes)
        src_title = up.name
        chunks = chunk_text(full_text, max_chars=chunk_size)
        for i, c in enumerate(chunks):
            meta = {"source": src_title, "chunk_id": f"{up.name}_chunk_{i+1}"}
            all_chunks.append(c)
            metadata.append(meta)
    if len(all_chunks) == 0:
        st.warning("No text found in uploaded PDFs.")
    else:
        with st.spinner("Embedding chunks and building index..."):
            embeddings = embed_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(embeddings)
            index = build_faiss_index(embeddings)
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            st.session_state.texts = all_chunks
            st.session_state.metadata = metadata
            st.session_state.index_built = True
        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

# Claim input
st.header("🔍 Enter Claim to Verify")
claim = st.text_area("Type your claim here (e.g. 'Green tea cures cancer')", height=100)

col1, col2 = st.columns([1,1])
with col1:
    verify_btn = st.button("✅ Verify Claim")
with col2:
    clear_history = st.button("🧹 Clear History")

if "history" not in st.session_state:
    st.session_state.history = []

if clear_history:
    st.session_state.history = []
    st.success("History cleared.")

if verify_btn:
    if not claim.strip():
        st.error("Please enter a claim.")
    elif not st.session_state.get("index_built", False):
        st.error("Please upload research paper PDFs first.")
    else:
        claim_emb = embed_model.encode([claim], convert_to_numpy=True)
        faiss.normalize_L2(claim_emb)
        D, I = st.session_state.index.search(claim_emb, top_k)

        retrieved = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            retrieved.append({
                "score": float(score),
                "text": st.session_state.texts[idx],
                "metadata": st.session_state.metadata[idx]
            })

        if not retrieved:
            st.warning("No relevant evidence found in uploaded papers.")
        else:
            evidence_texts = [r["text"] for r in retrieved]
            nli_results = run_nli_batch(evidence_texts, claim)
            combined = []
            for r, nli in zip(retrieved, nli_results):
                combined.append({
                    "text": r["text"],
                    "score": r["score"],
                    "metadata": r["metadata"],
                    "nli": nli
                })

            total_sim = sum([c["score"] for c in combined]) or 1e-6
            entail_weighted = sum([c["nli"]["entailment"] * c["score"] for c in combined]) / total_sim
            contra_weighted = sum([c["nli"]["contradiction"] * c["score"] for c in combined]) / total_sim
            neutral_weighted = sum([c["nli"]["neutral"] * c["score"] for c in combined]) / total_sim

            if entail_weighted > max(contra_weighted, neutral_weighted) and entail_weighted > 0.55:
                verdict = "✅ Supported"
                confidence = entail_weighted
            elif contra_weighted > max(entail_weighted, neutral_weighted) and contra_weighted > 0.55:
                verdict = "❌ Refuted"
                confidence = contra_weighted
            else:
                verdict = "⚪ Not Enough Evidence"
                confidence = max(entail_weighted, contra_weighted, neutral_weighted)

            st.session_state.history.insert(0, {
                "claim": claim,
                "verdict": verdict,
                "confidence": float(confidence),
                "evidence": combined
            })

            st.metric(label="Final Verdict", value=verdict, delta=f"{confidence:.2f}")
            st.write(f"**Claim:** {claim}")
            st.write(f"**Confidence Score:** {confidence:.3f}")

            # Pie Chart Visualization
            st.subheader("📊 Confidence Distribution")
            labels = ['Entailment (Supports)', 'Contradiction (Refutes)', 'Neutral']
            sizes = [entail_weighted, contra_weighted, neutral_weighted]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Summary Table
            st.subheader("Summary of Confidence Scores")
            summary_df = pd.DataFrame({
                'Type': ['Supported (Entailment)', 'Refuted (Contradiction)', 'Neutral'],
                'Confidence': [entail_weighted, contra_weighted, neutral_weighted]
            })
            st.table(summary_df)

            # Evidence
            st.markdown("### 🔍 Top Evidence from Papers")
            for i, c in enumerate(combined, start=1):
                st.markdown(f"**{i}. Source:** `{c['metadata']['source']}` — Similarity `{c['score']:.3f}`")
                probs = c["nli"]
                st.write(f"Entailment: {probs['entailment']:.3f} | Contradiction: {probs['contradiction']:.3f} | Neutral: {probs['neutral']:.3f}")
                snippet = c["text"]
                if len(snippet) > 800:
                    snippet = snippet[:800] + "..."
                st.info(snippet)
                st.write("---")

# History Section
if st.session_state.history:
    st.header("🕒 History of Claims")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df[['claim', 'verdict', 'confidence']])

    supported = sum("Supported" in h['verdict'] for h in st.session_state.history)
    refuted = sum("Refuted" in h['verdict'] for h in st.session_state.history)
    unknown = sum("Not Enough Evidence" in h['verdict'] for h in st.session_state.history)

    st.subheader("📈 Overall Summary")
    st.write(f"✅ Supported: {supported}")
    st.write(f"❌ Refuted: {refuted}")
    st.write(f"⚪ Not Enough Evidence: {unknown}")

    # Optional: simple summary pie chart
    labels = ['Supported', 'Refuted', 'Not Enough Evidence']
    sizes = [supported, refuted, unknown]
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)