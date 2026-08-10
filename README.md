# Claim Verifier AI (MythBuster)

An AI-driven claim verification and automated fact-checking system built using PyTorch and Hugging Face Transformers. The application uses a fine-tuned **RoBERTa-large-MNLI** model to evaluate textual claims against evidence retrieved from structured datasets like FEVER and SciFact.

---

## Features

- **Automated Claim Verification:** Analyzes user-submitted claims against verified dataset evidence.
- **Natural Language Inference (NLI):** Powered by `roberta-large-mnli` to classify relationships between claims and evidence (e.g., Entailment, Contradiction, Neutral).
- **Dataset Integration:** Uses pre-processed benchmark datasets (`FEVER` and `SciFact`) for evidence matching.
- **Interactive Interface:** Easy-to-use Python application interface (`app.py`).

---

## File Structure

```text
claimverifier/
│
├── data/
│   ├── fever dataset.zip      # FEVER benchmark dataset
│   └── scifact.zip            # SciFact benchmark dataset
│
├── app.py                     # Main application entry point
└── README.md                  # Project documentation# claimverifier
