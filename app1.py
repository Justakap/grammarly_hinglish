from pathlib import Path
import random
import re

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

# Minimum number of suggestions that should be returned
MIN_SUGGESTIONS = 3

# Maximum allowed numeric suggestion
MAX_ALLOWED_NUMBER = 3000


# ============================================================
# HINGLISH FALLBACK SUGGESTIONS
# ============================================================

HINGLISH_FALLBACKS = [
    "aree",
    "yaar",
    "haan",
    "acha",
    "accha",
    "arre",
    "bhai",
    "okay",
    "nahi",
    "toh",
    "kya",
    "bas",
    "sahi",
    "chalo",
    "phir",
    "matlab",
    "waise",
    "dekho",
    "sun",
    "haanji",
]


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


```python
# ============================================================
# CHECK IF WORD IS AN INVALID NUMBER
# ============================================================

def is_invalid_number(word: str) -> bool:

    """
    Returns True if the word is a number that should NOT
    be shown as a suggestion.

    Rules:
        - Numbers greater than 1000 are rejected.
        - Numbers having more than 4 digits are rejected.

    Examples:

        99       -> False  (allowed)
        999      -> False  (allowed)
        1000     -> False  (allowed)
        1001     -> True   (greater than 1000)
        9999     -> True   (4 digits but > 1000)
        10000    -> True   (5 digits)
        50000    -> True   (5 digits)
        1,500    -> True   (greater than 1000)
        1000.5   -> True   (greater than 1000)
    """

    cleaned = word.strip()

    # Remove commas from numbers such as:
    # 1,000
    # 10,000
    cleaned_without_commas = cleaned.replace(",", "")

    # --------------------------------------------------------
    # Check whether it is numeric
    # --------------------------------------------------------

    try:

        number = float(cleaned_without_commas)

    except ValueError:

        # Not a number -> valid Hinglish/English word
        return False

    # --------------------------------------------------------
    # Rule 1:
    # Reject numbers greater than 1000
    # --------------------------------------------------------

    if number > MAX_ALLOWED_NUMBER:
        return True

    # --------------------------------------------------------
    # Rule 2:
    # Reject numbers having more than 4 digits
    # --------------------------------------------------------

    # Remove decimal point and minus sign so that we can
    # count only numeric digits.
    digits_only = re.sub(
        r"[^0-9]",
        "",
        cleaned_without_commas
    )

    if len(digits_only) > 4:
        return True

    return False
```


# ============================================================
# CHECK IF SUGGESTION IS VALID
# ============================================================

def is_valid_suggestion(word: str) -> bool:

    if not word:
        return False

    word = word.strip()

    if not word:
        return False

    # Reject numbers > 1000
    if is_invalid_number(word):
        return False

    return True


# ============================================================
# ADD FALLBACK SUGGESTIONS
# ============================================================

def add_fallback_suggestions(
    predictions: list,
    seen: set,
    minimum: int = MIN_SUGGESTIONS
):

    """
    Adds random Hinglish words until the minimum number
    of suggestions is reached.
    """

    if len(predictions) >= minimum:
        return predictions

    # Make a copy so original list isn't modified
    available_fallbacks = [
        word
        for word in HINGLISH_FALLBACKS
        if word.lower() not in seen
    ]

    # Randomize fallback order
    random.shuffle(available_fallbacks)

    for word in available_fallbacks:

        if word.lower() in seen:
            continue

        predictions.append(word)

        seen.add(word.lower())

        if len(predictions) >= minimum:
            break

    return predictions


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    context: str,
    top_k: int = TOP_K
):

    context = context.strip()

    # No text -> no suggestions
    if not context:
        return []

    # --------------------------------------------------------
    # Keep top_k reasonable
    # --------------------------------------------------------

    # Always return at least 3 suggestions
    # Maximum 10 suggestions
    top_k = max(
        MIN_SUGGESTIONS,
        min(top_k, 10)
    )

    # --------------------------------------------------------
    # Generate more sequences than requested
    #
    # This is important because several generated sequences
    # can have the same first word.
    # --------------------------------------------------------

    num_return_sequences = max(
        NUM_BEAMS,
        top_k * 4
    )

    num_beams = num_return_sequences

    # ========================================================
    # TOKENIZE
    # ========================================================

    inputs = tokenizer(
        context,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    # ========================================================
    # GENERATE
    # ========================================================

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

    # ========================================================
    # EXTRACT UNIQUE FIRST WORDS
    # ========================================================

    predictions = []

    seen = set()

    for output in outputs:

        # ----------------------------------------------------
        # Get only newly generated tokens
        # ----------------------------------------------------

        generated_ids = output[input_length:]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        if not generated_text:
            continue

        # ----------------------------------------------------
        # Get first generated word
        # ----------------------------------------------------

        words = generated_text.split()

        if not words:
            continue

        word = words[0].strip()

        # ----------------------------------------------------
        # Ignore empty candidates
        # ----------------------------------------------------

        if not word:
            continue

        # ----------------------------------------------------
        # Ignore invalid numbers
        # ----------------------------------------------------

        if not is_valid_suggestion(word):
            continue

        # ----------------------------------------------------
        # Remove duplicates
        #
        # Case insensitive:
        #
        # "Yaar"
        # "yaar"
        #
        # are considered the same.
        # ----------------------------------------------------

        normalized_word = word.lower()

        if normalized_word in seen:
            continue

        seen.add(normalized_word)

        predictions.append(word)

        # ----------------------------------------------------
        # Stop once we have enough suggestions
        # ----------------------------------------------------

        if len(predictions) >= top_k:
            break

    # ========================================================
    # ADD FALLBACKS IF LESS THAN 3 SUGGESTIONS
    # ========================================================

    predictions = add_fallback_suggestions(
        predictions,
        seen,
        MIN_SUGGESTIONS
    )

    # ========================================================
    # FINAL SAFETY FILTER
    # ========================================================

    # This makes absolutely sure that no number > 1000
    # reaches the frontend, including fallback suggestions.
    final_predictions = []

    for word in predictions:

        if not is_valid_suggestion(word):
            continue

        normalized_word = word.lower()

        if normalized_word in {
            item.lower()
            for item in final_predictions
        }:
            continue

        final_predictions.append(word)

        if len(final_predictions) >= top_k:
            break

    # ========================================================
    # EXTRA FALLBACK
    # ========================================================

    # Normally the previous fallback guarantees 3 suggestions.
    # This is an additional safety net.
    if len(final_predictions) < MIN_SUGGESTIONS:

        seen_final = {
            word.lower()
            for word in final_predictions
        }

        final_predictions = add_fallback_suggestions(
            final_predictions,
            seen_final,
            MIN_SUGGESTIONS
        )

    return final_predictions[:top_k]


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


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "device": DEVICE,

        "base_model": BASE_MODEL,

        "adapter": str(ADAPTER_PATH),
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict_api(
    request: PredictRequest
):

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

    print("\nStarting local server...")

    print("http://127.0.0.1:8000")

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000
    )
