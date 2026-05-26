# NLP Healthcare Document Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Azure](https://img.shields.io/badge/Azure-Cognitive%20Services-0089D6) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project builds an intelligent **NLP-driven document classification system** for enterprise healthcare environments. It automatically classifies incoming clinical documents — insurance claims, discharge summaries, lab reports, and referral letters — routing them to the correct processing queues without any manual intervention.

The system cut manual claims processing effort by approximately **41%** in our test environment by eliminating the need for staff to read and hand-route each incoming document.

---

## The Problem It Solves

Healthcare organizations receive thousands of unstructured documents daily. Without automation, staff spend hours triaging them. This system takes a scanned or digital document, extracts the text, and classifies it into one of several clinical categories in under 200ms.

---

## Architecture

```
Incoming Documents (PDF / DOCX / Scanned Images)
        |
        v
  Azure Cognitive Services (OCR + Text Extraction)
        |
        v
  Text Preprocessing & Tokenization (HuggingFace)
        |
        v
  Fine-tuned BERT Classifier (PyTorch)
        |
        v
  Classification Output + Confidence Score
        |
        v
  Routing Engine → Processing Queue
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Text Extraction | Azure Cognitive Services (Form Recognizer) |
| NLP Model | HuggingFace Transformers (BERT, DistilBERT) |
| Deep Learning | PyTorch 2.0 |
| Serving | FastAPI + Docker |
| Training Infrastructure | Azure Machine Learning |
| Experiment Tracking | MLflow |
| Storage | Azure Blob Storage |

---

## Document Categories

The classifier handles these healthcare document types:

- **Insurance Claims** — EOB forms, pre-authorization requests
- - **Discharge Summaries** — Post-hospitalization patient records
  - - **Lab Reports** — Blood panels, pathology, imaging results
    - - **Referral Letters** — Specialist referrals and follow-ups
      - - **Prescription Orders** — Medication requests and renewals
        - - **Operative Notes** — Surgical procedure documentation
         
          - ---

          ## Project Structure

          ```
          nlp-healthcare-document-classifier/
          ├── data/
          │   ├── raw/                         # Raw document samples
          │   └── labeled/                     # Annotated training dataset
          ├── notebooks/
          │   ├── 01_data_exploration.ipynb
          │   ├── 02_fine_tuning_bert.ipynb
          │   └── 03_evaluation_metrics.ipynb
          ├── src/
          │   ├── extraction/
          │   │   └── azure_ocr.py             # Azure Form Recognizer integration
          │   ├── preprocessing/
          │   │   └── text_cleaner.py          # Text normalization pipeline
          │   ├── model/
          │   │   ├── fine_tune.py             # BERT fine-tuning script
          │   │   └── classifier.py            # Inference logic
          │   ├── api/
          │   │   └── main.py                  # FastAPI endpoint
          │   └── utils/
          │       └── metrics.py               # Evaluation utilities
          ├── docker/
          │   └── Dockerfile
          ├── azure_ml/
          │   └── train_config.yml
          ├── tests/
          │   └── test_classifier.py
          ├── requirements.txt
          └── README.md
          ```

          ---

          ## Model Performance

          | Class | Precision | Recall | F1-Score |
          |---|---|---|---|
          | Insurance Claims | 0.94 | 0.91 | 0.92 |
          | Discharge Summaries | 0.89 | 0.93 | 0.91 |
          | Lab Reports | 0.96 | 0.94 | 0.95 |
          | Referral Letters | 0.88 | 0.86 | 0.87 |
          | **Overall** | **0.92** | **0.91** | **0.91** |

          ---

          ## Quickstart

          ```bash
          git clone https://github.com/saigopichowdaryvadde-cpu/nlp-healthcare-document-classifier.git
          cd nlp-healthcare-document-classifier

          pip install -r requirements.txt

          # Fine-tune the BERT model on your dataset
          python src/model/fine_tune.py --data-dir data/labeled --epochs 5

          # Start the classification API
          uvicorn src.api.main:app --host 0.0.0.0 --port 8000

          # Test with a sample document
          curl -X POST "http://localhost:8000/classify" \
            -H "Content-Type: application/json" \
            -d '{"text": "Patient was discharged following appendectomy..."}'
          ```

          ---

          ## Sample API Response

          ```json
          {
            "document_id": "doc_20240315_001",
            "predicted_class": "Discharge Summary",
            "confidence": 0.97,
            "processing_queue": "clinical-records",
            "latency_ms": 143
          }
          ```

          ---

          ## Notes & Lessons Learned

          Fine-tuning BERT on domain-specific healthcare text was surprisingly different from general NLP tasks. Medical abbreviations and clinical shorthand were a real challenge — I ended up building a custom preprocessing step to expand common clinical abbreviations before tokenization. That alone improved F1 by about 4 points.

          Azure Form Recognizer handled scanned PDFs far better than open-source OCR alternatives we tried initially.

          ---

          ## License

          MIT License — see [LICENSE](LICENSE) for details.
