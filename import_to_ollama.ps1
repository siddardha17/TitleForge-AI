# TitleForge AI — Import fine-tuned GGUF into Ollama
# Run this AFTER downloading titleforge-q4_K_M.gguf from Google Colab

$GGUF_DEST = "f:\Gen-Ai\TitleForge-AI\export\gguf"
$GGUF_FILE = "$GGUF_DEST\titleforge-q4_K_M.gguf"
$MODELFILE  = "f:\Gen-Ai\TitleForge-AI\export\Modelfile"
$MODEL_NAME = "titleforge"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TitleForge AI — Ollama Import Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Step 1: Create gguf directory
if (-not (Test-Path $GGUF_DEST)) {
    New-Item -ItemType Directory -Path $GGUF_DEST | Out-Null
    Write-Host "[OK] Created directory: $GGUF_DEST" -ForegroundColor Green
}

# Step 2: Check GGUF file exists
if (-not (Test-Path $GGUF_FILE)) {
    Write-Host ""
    Write-Host "[ERROR] GGUF file not found at:" -ForegroundColor Red
    Write-Host "  $GGUF_FILE" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please move your downloaded GGUF file here:" -ForegroundColor White
    Write-Host "  $GGUF_DEST\" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Then re-run this script." -ForegroundColor White
    exit 1
}

$size = (Get-Item $GGUF_FILE).Length / 1MB
Write-Host "[OK] Found GGUF: $([math]::Round($size,1)) MB" -ForegroundColor Green

# Step 3: Write Modelfile with absolute path
$modelfileContent = @"
FROM $GGUF_FILE

SYSTEM """You are TitleForge AI, an expert e-commerce product title normalization assistant.

Your ONLY task is to convert a raw, messy vendor product title into a clean, well-formatted, standardized catalog title.

Rules you MUST follow:
- Apply proper Title Case to brand names, product names, and key attributes
- Expand common abbreviations (blk=Black, wrls=Wireless, sz10=Size 10, hdphn=Headphones, noisecancelling=Noise Cancelling, battry=Battery, gen=Generation)
- Remove noise words and promotional tags (e.g. !!SALE!!, ***HOT***, BUY NOW)
- Preserve all technical specifications (storage, RAM, screen size, color, connectivity)
- Correct spacing and punctuation
- Output ONLY the clean title with no explanation

Respond with the normalized title and nothing else."""

PARAMETER temperature 0.05
PARAMETER num_predict 80
PARAMETER repeat_penalty 1.1
PARAMETER top_p 0.9
PARAMETER stop "\n"
PARAMETER stop "###"
"@

$modelfileContent | Out-File -Encoding utf8 -FilePath "$GGUF_DEST\Modelfile.generated"
Write-Host "[OK] Generated Modelfile at $GGUF_DEST\Modelfile.generated" -ForegroundColor Green

# Step 4: Create Ollama model
Write-Host ""
Write-Host "[...] Creating Ollama model '$MODEL_NAME'..." -ForegroundColor Cyan
ollama create $MODEL_NAME -f "$GGUF_DEST\Modelfile.generated"
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Model '$MODEL_NAME' created in Ollama!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] ollama create failed. Check the output above." -ForegroundColor Red
    exit 1
}

# Step 5: Quick test
Write-Host ""
Write-Host "[...] Running quick test..." -ForegroundColor Cyan
$testPrompt = "### Instruction:`nNormalize this raw product title into a clean, standardized catalog title. Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting.`n`n### Input:`nAPPLE IPHONE 15 PRO MAX 256GB BLK`n`n### Response:`n"
$result = $testPrompt | ollama run $MODEL_NAME
Write-Host "  Input : APPLE IPHONE 15 PRO MAX 256GB BLK" -ForegroundColor White
Write-Host "  Output: $result" -ForegroundColor Green

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DONE! Next steps:" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Start API server (Ollama mode):" -ForegroundColor White
Write-Host '     cd f:\Gen-Ai\TitleForge-AI' -ForegroundColor Yellow
Write-Host '     $env:MODEL_BACKEND="ollama"; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000' -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Open browser: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "  3. Test API:" -ForegroundColor White
Write-Host '     curl -X POST http://localhost:8000/normalize -H "Content-Type: application/json" -d "{\"raw_title\": \"APPLE IPHONE 15 PRO MAX 256GB BLK\"}"' -ForegroundColor Yellow
Write-Host ""
