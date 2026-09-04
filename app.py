from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"

PROJECT_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = PROJECT_DIR / "model"
INDEX_PATH = PROJECT_DIR / "index.html"

TOP_K = 5
NUM_BEAMS = 20
MAX_NEW_TOKENS = 8


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print("=" * 60)
print("HINGLISH NEXT-WORD PREDICTOR")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Adapter: {ADAPTER_PATH}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

print("✅ Tokenizer loaded")


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading Qwen base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=DTYPE,
)

base_model = base_model.to(DEVICE)

print("✅ Base model loaded")


# ============================================================
# LOAD YOUR TRAINED LORA ADAPTER
# ============================================================

print("\nLoading trained Hinglish adapter...")

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"\n❌ Adapter folder not found:\n{ADAPTER_PATH}\n\n"
        "Make sure the contents of qwen3-0.6b-hinglish-lora-final "
        "are inside the 'model' folder."
    )

if not (ADAPTER_PATH / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"\n❌ adapter_config.json not found in:\n{ADAPTER_PATH}"
    )

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH)
)

model.eval()

print("✅ Trained Hinglish adapter loaded")
print("=" * 60)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hinglish Next Word Predictor"
)


# ============================================================
# REQUEST FORMAT
# ============================================================

class PredictRequest(BaseModel):
    text: str
    top_k: int = TOP_K


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(context: str, top_k: int = TOP_K):

    context = context.strip()

    if not context:
        return []

    # Keep top_k reasonable
    top_k = max(1, min(top_k, 10))

    # We generate more sequences than requested because
    # several generations can have the same first word.
    num_return_sequences = max(
        NUM_BEAMS,
        top_k * 4
    )

    num_beams = num_return_sequences

    # Tokenize
    inputs = tokenizer(
        context,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    # Generate
    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            num_beams=num_beams,
            num_return_sequences=num_return_sequences,

            do_sample=False,

            early_stopping=True,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract unique first words
    predictions = []
    seen = set()

    for output in outputs:

        generated_ids = output[input_length:]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        if not generated_text:
            continue

        words = generated_text.split()

        if not words:
            continue

        word = words[0].strip()

        # Ignore empty / duplicate candidates
        if not word:
            continue

        if word in seen:
            continue

        seen.add(word)
        predictions.append(word)

        if len(predictions) >= top_k:
            break

    return predictions


# ============================================================
# WEB ROUTES
# ============================================================

@app.get("/")
def home():

    if not INDEX_PATH.exists():
        return {
            "error": "index.html not found",
            "expected": str(INDEX_PATH)
        }

    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": DEVICE,
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER_PATH),
    }


@app.post("/predict")
def predict_api(request: PredictRequest):

    predictions = predict(
        request.text,
        request.top_k
    )

    return {
        "predictions": predictions
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n Starting local server...")
    print("http://127.0.0.1:8000")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )