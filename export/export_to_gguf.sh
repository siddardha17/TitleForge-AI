#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  TitleForge AI — Export Fine-Tuned Model to GGUF for Ollama
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Prerequisites (run once):
#    1. git clone https://github.com/ggerganov/llama.cpp
#    2. cd llama.cpp && make -j$(nproc)
#    3. pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
#
#  Usage:
#    chmod +x export_to_gguf.sh
#    ./export_to_gguf.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MERGED_MODEL="$PROJECT_ROOT/output/merged-model"
EXPORT_DIR="$PROJECT_ROOT/export/gguf"
LLAMA_CPP_DIR="$PROJECT_ROOT/llama.cpp"        # Change if installed elsewhere
QUANT_TYPE="q4_K_M"                             # Quantization level
MODEL_NAME="titleforge"

# ── Color helpers ──────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Step 1: Validate merged model exists ───────────────────────────────────────
info "Checking merged model at: $MERGED_MODEL"
[ -d "$MERGED_MODEL" ] || error "Merged model not found. Run training/finetune.py first!"

# ── Step 2: Clone llama.cpp if not present ─────────────────────────────────────
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    info "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
    make -j"$(nproc)"
    pip install -r requirements/requirements-convert_hf_to_gguf.txt
    cd "$SCRIPT_DIR"
else
    info "llama.cpp already present at: $LLAMA_CPP_DIR"
fi

# ── Step 3: Create output directory ───────────────────────────────────────────
mkdir -p "$EXPORT_DIR"

# ── Step 4: Convert HuggingFace model → GGUF (F16) ───────────────────────────
info "Converting HuggingFace model to GGUF (F16)..."
GGUF_F16="$EXPORT_DIR/${MODEL_NAME}-f16.gguf"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_MODEL" \
    --outtype f16 \
    --outfile "$GGUF_F16"

info "✅ F16 GGUF saved → $GGUF_F16"

# ── Step 5: Quantize to Q4_K_M (recommended for local use) ───────────────────
info "Quantizing to ${QUANT_TYPE}..."
GGUF_QUANT="$EXPORT_DIR/${MODEL_NAME}-${QUANT_TYPE}.gguf"

"$LLAMA_CPP_DIR/llama-quantize" \
    "$GGUF_F16" \
    "$GGUF_QUANT" \
    "$QUANT_TYPE"

info "✅ Quantized GGUF saved → $GGUF_QUANT"

# ── Step 6: Update Modelfile with correct GGUF path ───────────────────────────
MODELFILE="$SCRIPT_DIR/Modelfile"
sed -i "s|FROM .*|FROM $GGUF_QUANT|" "$MODELFILE"
info "✅ Modelfile updated → $MODELFILE"

# ── Step 7: Load model into Ollama ────────────────────────────────────────────
info "Creating Ollama model '${MODEL_NAME}'..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ TitleForge model is ready in Ollama!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "  Test it with:"
echo "  ollama run ${MODEL_NAME}"
echo ""
echo "  Or send a prompt directly:"
echo "  ollama run ${MODEL_NAME} \"### Instruction:"
echo "  Normalize this raw product title into a clean, standardized catalog title."
echo "  Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting."
echo ""
echo "  ### Input:"
echo "  APPLE IPHONE 15 PRO MAX 256GB BLK"
echo ""
echo "  ### Response:\""
