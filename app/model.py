"""
TitleForge AI — Dual-Backend Model Loader
Supports: local HuggingFace merged model  OR  Ollama REST API
Configured via MODEL_BACKEND env var: "local" | "ollama"
"""

import os
import time
import logging
import requests
import torch
from abc import ABC, abstractmethod
from typing import Optional

from app.utils import preprocess, postprocess, rule_based_normalize

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_BACKEND   = os.getenv("MODEL_BACKEND", "local")   # "local" or "ollama"
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "merged-model"
))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "titleforge")
MAX_NEW_TOKENS  = 80

INFERENCE_TEMPLATE = (
    "### Instruction:\n"
    "Normalize this raw product title into a clean, standardized catalog title. "
    "Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting.\n\n"
    "### Input:\n{raw_title}\n\n"
    "### Response:\n"
)


# ── Abstract base ──────────────────────────────────────────────────────────────
class BaseModel(ABC):
    @abstractmethod
    def normalize(self, raw_title: str) -> str: ...
    @abstractmethod
    def is_ready(self) -> bool: ...
    @property
    @abstractmethod
    def info(self) -> dict: ...


# ── Local HuggingFace Model ────────────────────────────────────────────────────
class LocalModel(BaseModel):
    def __init__(self):
        self._pipe = None
        self._device = None
        self._model_id = LOCAL_MODEL_DIR

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        if not os.path.isdir(LOCAL_MODEL_DIR):
            logger.warning(
                f"Merged model not found at {LOCAL_MODEL_DIR}. "
                "Falling back to rule-based normalization."
            )
            return

        logger.info(f"Loading local model from {LOCAL_MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
        logger.info(f"✅ Local model loaded on {self._device}")

    def normalize(self, raw_title: str) -> str:
        cleaned = preprocess(raw_title)
        if self._pipe is None:
            logger.warning("Local model not loaded — using rule-based fallback.")
            return rule_based_normalize(raw_title)

        prompt = INFERENCE_TEMPLATE.format(raw_title=cleaned)
        output = self._pipe(prompt)[0]["generated_text"]
        response = output.split("### Response:\n")[-1].strip()
        return postprocess(response.split("\n")[0].strip())

    def is_ready(self) -> bool:
        return self._pipe is not None

    @property
    def info(self) -> dict:
        return {
            "model_backend": "local",
            "model_id": self._model_id,
            "device": self._device or "not loaded",
        }


# ── Ollama Model ───────────────────────────────────────────────────────────────
class OllamaModel(BaseModel):
    def __init__(self):
        self._ready = False
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if any(OLLAMA_MODEL in m for m in models):
                    self._ready = True
                    logger.info(f"✅ Ollama model '{OLLAMA_MODEL}' is ready.")
                else:
                    logger.warning(
                        f"Ollama is running but model '{OLLAMA_MODEL}' not found. "
                        f"Available: {models}. Run 'ollama create titleforge -f export/Modelfile'."
                    )
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable. Is it running? Start with: ollama serve")

    def normalize(self, raw_title: str) -> str:
        cleaned = preprocess(raw_title)
        prompt = INFERENCE_TEMPLATE.format(raw_title=cleaned)

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "num_predict": MAX_NEW_TOKENS,
                "stop": ["\n", "###"],
            },
        }

        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            response_text = r.json().get("response", "").strip()
            return postprocess(response_text.split("\n")[0].strip())
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return rule_based_normalize(raw_title)

    def is_ready(self) -> bool:
        return self._ready

    @property
    def info(self) -> dict:
        return {
            "model_backend": "ollama",
            "ollama_model": OLLAMA_MODEL,
            "ollama_url": OLLAMA_BASE_URL,
        }


# ── Factory ────────────────────────────────────────────────────────────────────
_model_instance: Optional[BaseModel] = None


def get_model() -> BaseModel:
    """Return the singleton model instance (lazy-loaded)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = _create_model()
    return _model_instance


def _create_model() -> BaseModel:
    backend = MODEL_BACKEND.lower()
    logger.info(f"Initializing model backend: '{backend}'")

    if backend == "ollama":
        return OllamaModel()
    else:
        m = LocalModel()
        m.load()
        return m
