# ⚒️ TitleForge AI v2 — Product Title Normalization Engine

> **Fine-tune a language model to convert messy vendor product titles into clean, standardized catalog titles — then deploy locally with Ollama.**

---

## 📁 Project Structure

```
TitleForge-AI/
├── data/
│   ├── sample_dataset.csv          # 150+ raw→clean training pairs
│   └── prepare_data.py             # Dataset builder & train/val/test splitter
├── training/
│   ├── finetune.py                 # QLoRA fine-tuning (TinyLlama + LoRA/PEFT)
│   └── evaluate.py                 # BLEU, ROUGE, Exact Match, CER metrics
├── export/
│   ├── export_to_gguf.sh           # GGUF conversion script for Ollama
│   └── Modelfile                   # Ollama model configuration
├── app/
│   ├── main.py                     # FastAPI application
│   ├── model.py                    # Dual-backend: local HF model or Ollama
│   ├── utils.py                    # Text pre/post-processing
│   └── schemas.py                  # Pydantic request/response models
├── frontend/
│   ├── index.html                  # Premium glassmorphic UI
│   ├── styles.css                  # Dark-mode design system
│   └── script.js                   # API integration + UI logic
├── notebooks/
│   └── TitleForge_Colab.ipynb      # Full Google Colab training notebook
├── output/                         # Created after training
│   ├── lora-adapter/               # LoRA adapter weights
│   └── merged-model/               # Full merged HuggingFace model
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python data/prepare_data.py
```

This reads `data/sample_dataset.csv` and outputs train/val/test splits to `data/processed/`.

### 3. Fine-Tune the Model (Local GPU)

```bash
python training/finetune.py
```

> ⚠️ Requires a CUDA-capable GPU with at least 8GB VRAM.
> For CPU-only or free cloud training, use the **Google Colab notebook** instead.

### 4. Evaluate the Model

```bash
python training/evaluate.py
```

Reports BLEU, ROUGE-1/2/L, Exact Match %, and Character Error Rate.

### 5. Start the FastAPI Server

```bash
# Default: local HuggingFace model
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use Ollama backend (after export):
MODEL_BACKEND=ollama uvicorn app.main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## 🧠 Training on Google Colab (Recommended)

1. Open `notebooks/TitleForge_Colab.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change Runtime Type)
3. Run all cells step by step
4. Download the trained GGUF file when done

---

## 🦙 Ollama Deployment

### Prerequisites

- [Ollama](https://ollama.ai/download) installed and running
- `llama.cpp` (auto-cloned by the export script)

### Export & Load

```bash
# Run the export script (Linux/Mac/WSL)
chmod +x export/export_to_gguf.sh
./export/export_to_gguf.sh
```

This will:
1. Convert the merged HuggingFace model → F16 GGUF
2. Quantize to Q4_K_M (~700MB)
3. Auto-load into Ollama as `titleforge`

### Test Ollama Inference

```bash
ollama run titleforge "### Instruction:
Normalize this raw product title into a clean, standardized catalog title. Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting.

### Input:
APPLE IPHONE 15 PRO MAX 256GB BLK

### Response:"
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/normalize` | Normalize a single raw title |
| `POST` | `/bulk-normalize` | Upload CSV → download normalized CSV |
| `GET`  | `/health` | Service & model health check |
| `GET`  | `/model-info` | Current model backend info |
| `GET`  | `/docs` | Swagger UI (auto-generated) |

### Example — Single Normalization

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_title": "APPLE IPHONE 15 PRO MAX 256GB BLK"}'
```

**Response:**
```json
{
  "raw_title": "APPLE IPHONE 15 PRO MAX 256GB BLK",
  "normalized_title": "Apple iPhone 15 Pro Max 256GB Black",
  "model_backend": "local",
  "processing_time_ms": 142.5
}
```

### Example — Bulk Normalization

```bash
curl -X POST http://localhost:8000/bulk-normalize \
  -F "file=@data/sample_dataset.csv" \
  --output normalized_output.csv
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BACKEND` | `local` | `local` or `ollama` |
| `LOCAL_MODEL_DIR` | `output/merged-model` | Path to merged HF model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `titleforge` | Ollama model name |

---

## 🐳 Docker

```bash
# Build (after training — requires output/merged-model/)
docker build -t titleforge-ai .

# Run
docker run -p 8000:8000 -e MODEL_BACKEND=local titleforge-ai
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU** | Bilingual Evaluation Understudy — n-gram overlap |
| **ROUGE-1/2/L** | Recall-Oriented Understudy for Gisting Evaluation |
| **Exact Match** | % of predictions matching gold exactly (case-insensitive) |
| **CER** | Character Error Rate — edit distance at character level |

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Base Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Fine-Tuning Method | QLoRA (4-bit NF4 + LoRA rank 16) |
| Training Framework | `trl` SFTTrainer |
| GGUF Quantization | Q4_K_M |
| Approx. GGUF Size | ~700 MB |
| Prompt Format | Alpaca-style instruction format |

---

## 🛠 Tech Stack

- **Python 3.11**
- **HuggingFace Transformers** — model loading & inference
- **PEFT + TRL** — LoRA/QLoRA fine-tuning
- **bitsandbytes** — 4-bit NF4 quantization
- **FastAPI + Uvicorn** — REST API
- **llama.cpp** — GGUF conversion & quantization
- **Ollama** — local model serving
- **Vanilla HTML/CSS/JS** — frontend UI
