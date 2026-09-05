# Hinglish Next-Word Predictor

A real-time **Hinglish next-word prediction system** designed to suggest the next word while a user is typing.

The project combines **Qwen3-0.6B-Base** with **LoRA fine-tuning** on a Hinglish corpus and provides a lightweight local browser interface.

---

# 1. Introduction

## 1.1 Motivation

Hinglish is widely used for communication in which Hindi and English are mixed, often using Roman script. Examples include _“aap kya kar rahe ho”_, _“mujhe kal office jana hai”_, and _“main ghar ja raha”_. General-purpose language models and conventional next-word predictors may struggle with Romanized Hindi, code-mixing, informal spelling, and context-dependent language switching. Solving this problem has real-world implications for keyboards, messaging applications, writing assistants, search systems, and accessibility tools. In the future, the system could become a Grammarly-like writing assistant or intelligent keyboard specifically designed for Indian multilingual communication. If the project fails, users may continue receiving irrelevant or grammatically inappropriate suggestions when typing Hinglish.

## 1.2 Relation with NLP

This is an NLP problem because the system must understand the context of a partially completed Hinglish sentence and predict an appropriate continuation. It must model relationships between Hindi and English words, Romanized Hindi spelling patterns, sentence context, and code-switching behavior. The project also investigates text normalization, where noisy or non-standard Hinglish expressions can be mapped to more standardized forms before prediction. The normalization research corpus contains 13,494 Hindi-English code-mixed sentence pairs.

## 1.3 Problem Type

The primary task is **contextual next-word prediction**, a text-generation/language-modeling problem. Given a sequence such as:

```text
aap kya kar rahe
```

the model predicts likely next words. The underlying model is a causal language model. A secondary task is Hinglish text normalization, which is treated as a sequence-to-sequence text transformation problem.

---

# 2. Related Work

## 2.1 Hinglish Normalization

A major research basis for the normalization component is **“A Dataset for Hindi-English Code-Mixed Text Normalization.”**

The work introduces **13,494 Hindi-English code-mixed sentence pairs** containing non-standard input text and human-annotated normalized text.

The corpus includes:

- Short forms
- Acronyms
- Typos
- Wordplay
- Romanized Hindi
- Word splitting
- Word merging

The paper reports that **80.08% of sentences were modified after annotation**.

| Split    | Instances |
| -------- | --------: |
| Training |    10,795 |
| Test     |     2,699 |
| Total    |    13,494 |

The paper uses a **BiLSTM with attention** baseline.

Reported results:

| Metric | Result |
| ------ | -----: |
| WER    |  15.55 |
| BLEU   |  71.21 |
| METEOR |   0.50 |

The work also shows that normalization is context-dependent; an informal form can have different normalized outputs depending on its surrounding language context.

## 2.2 Baseline

A traditional **n-gram language model** was implemented for next-word prediction.

A backoff strategy was investigated:

```text
4-gram
  ↓
3-gram
  ↓
2-gram
  ↓
unigram
```

The baseline showed substantial sparsity for exact contexts. This motivated the use of a pretrained causal language model.

## 2.3 Proposed Model

The primary model is **Qwen/Qwen3-0.6B-Base**, adapted using **Low-Rank Adaptation (LoRA)**.

```text
r = 16
lora_alpha = 32
lora_dropout = 0.05
bias = none
task_type = CAUSAL_LM

target_modules:
    q_proj
    k_proj
    v_proj
    o_proj
```

LoRA allows the model to be adapted without full parameter fine-tuning.

---

# 3. Datasets

## 3.1 Primary Next-Word Dataset

The main dataset is:

```text
all.txt
```

Its structure is:

```text
token<TAB>language
```

Blank lines represent sentence boundaries.

Dataset statistics:

```text
1,355,497 total tokens
75,910 unique tokens
44,453 sentence blocks
```

After duplicate sentence removal:

```text
43,922 unique sentence strings
```

Final split:

| Split      | Instances |
| ---------- | --------: |
| Training   |    35,137 |
| Validation |     4,392 |
| Test       |     4,393 |
| Total      |    43,922 |

The split is performed at sentence level and duplicate sentence strings were removed before splitting to reduce sentence leakage.

## 3.2 Sentence Statistics

| Statistic            |       Value |
| -------------------- | ----------: |
| Mean sentence length | 30.49 words |
| Median               |    28 words |
| Minimum              |     7 words |
| Maximum              |    77 words |

Qwen tokenizer statistics on a training sample:

| Statistic       | Value |
| --------------- | ----: |
| Average tokens  | 49.65 |
| Median          |    45 |
| 95th percentile |    93 |
| 99th percentile |   100 |
| Maximum         |   109 |

A maximum sequence length of **128 tokens** was therefore selected.

## 3.3 Labels

The main dataset is not a conventional classification dataset. It is sequential text from which next-word targets are derived.

Example:

```text
Input:
aap kya kar rahe

Target:
ho
```

The original language tags identify Hindi/English tokens, but the prediction task is formulated as causal language modeling.

## 3.4 Dataset Bias

Word frequencies are not uniformly distributed. Frequent words include:

```text
apne
hai
ko
ki
ke
se
to
ka
me
bhi
```

Therefore the model may learn frequent words more strongly than rare words. Bias may also occur toward common conversational patterns, Romanized Hindi spellings, English words, and sentence structures represented in the corpus.

## 3.5 Normalization Dataset

A separate dataset:

```text
hinglishNorm.json
```

contains:

```text
13,494 input → normalized sentence pairs
10,795 training pairs
2,699 test pairs
```

It is **not currently part of the Qwen next-word model training data** and is intended for a separate internal normalization component.

## 3.6 HinGE

The project also investigated HinGE:

```text
1,976 parallel pairs
4,803 human-generated Hinglish sentences
3,952 machine-generated sentences
```

The human-generated portion may be useful for future augmentation/evaluation. Machine-generated data may introduce additional noise and is not used as the core training corpus.

---

# 4. Experimental Plan

## 4.1 Experimental Setting

```text
Base model:
Qwen/Qwen3-0.6B-Base

Fine-tuning:
LoRA

Task:
Causal language modeling / next-word prediction

Maximum sequence length:
128 tokens
```

The data is split into training, validation, and test sets with no overlap between unique sentence strings.

## 4.2 Training Configuration

```text
Epochs:                    2
Learning rate:             1e-4
Per-device batch size:     2
Gradient accumulation:     8
Maximum sequence length:   128
Gradient checkpointing:    enabled
FP16:                      enabled
```

Training was performed on a Google Colab Tesla T4.

Final run:

```text
35,137 training examples
4,392 validation examples
1,738 training steps
2 epochs
```

## 4.3 Hyperparameter Selection

The configuration was selected based on GPU memory, tokenizer sequence-length statistics, LoRA efficiency, a preliminary dry run, and validation loss.

Future experiments can compare:

```text
Learning rate
LoRA rank
LoRA alpha
Dropout
Sequence length
Batch size
```

A small controlled search or Bayesian/Optuna-style search is more practical than a large grid search because of computational cost.

## 4.4 Evaluation Metrics

The primary metrics are:

- **Top-1 accuracy**
- **Top-3 accuracy**
- **Top-5 accuracy**

These are appropriate because the final application presents multiple candidate suggestions.

Additional metrics:

```text
MRR
Perplexity
Token accuracy
Latency per prediction
```

Latency is particularly important for a real-time writing assistant.

## 4.5 Current Evaluation

An initial generation-based test evaluation produced:

```text
Top-1: 12.36%
Top-3: 19.28%
Top-5: 22.81%
Average: 0.543 seconds/example
```

These results are preliminary because Qwen uses subword tokenization while the evaluation extracts whitespace-delimited words from generated continuations. A more rigorous word-level candidate-ranking evaluation is therefore required for the final benchmark.

## 4.6 Final Demo

The final demonstration is a **live local web application**.

Example interaction:

```text
User:
aap kya kar

        ↓ press space

Suggestions:
[rahe] [ho] [hain] ...
```

Architecture:

```text
Browser UI
    ↓
FastAPI backend
    ↓
Qwen3-0.6B + LoRA
    ↓
Candidate generation
    ↓
Word filtering/ranking
    ↓
Suggestions
```

The application is designed to run locally on an Apple Silicon Mac using MPS.

---

# 5. Project Management

## 5.1 Gantt Chart

| Phase                   | W1  | W2  | W3  | W4  | W5  | W6  | W7  | W8  |
| ----------------------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Literature Review       |  █  |  █  |     |     |     |     |     |     |
| Dataset Analysis        |  █  |  █  |     |     |     |     |     |     |
| Preprocessing           |     |  █  |  █  |     |     |     |     |     |
| Baseline Model          |     |     |  █  |  █  |     |     |     |     |
| Qwen + LoRA Training    |     |     |     |  █  |  █  |     |     |     |
| Model Evaluation        |     |     |     |     |  █  |  █  |     |     |
| Normalization Component |     |     |     |     |     |  █  |     |     |
| UI / Backend            |     |     |     |     |     |  █  |  █  |     |
| Final Testing           |     |     |     |     |     |     |  █  |  █  |
| Documentation / Demo    |     |     |     |     |     |     |     |  █  |

## 5.2 Computational Resources

### Training

```text
GPU: NVIDIA Tesla T4
GPU memory: approximately 15 GB
Framework: PyTorch
Environment: Google Colab
```

LoRA was selected partly because full fine-tuning would require significantly more computational resources.

### Local Inference

```text
Machine: Apple Silicon Mac M3
Acceleration: Apple MPS
```

The final LoRA adapter is approximately:

```text
~17.5 MB
```

The larger Qwen base model can be downloaded and cached separately rather than committed to GitHub.

---

# 5.3 Project Success Criteria

The project is considered successful if:

### Model

1. The model produces meaningful Hinglish continuations.
2. Top-3 and Top-5 accuracy demonstrate useful ranking.
3. Performance is better than a simple n-gram baseline.
4. The model handles Hindi and English in the same context.

### Application

1. Users can type Hinglish naturally.
2. Suggestions are generated after a space.
3. Suggestions appear within acceptable latency.
4. Suggestions can be selected with mouse or keyboard.
5. The model can run locally without manually executing prediction code.

### Research

1. Preprocessing is documented.
2. Training configuration is reproducible.
3. Evaluation methodology is clearly defined.
4. Limitations of word-level evaluation are documented.
5. Normalization is clearly separated from the current Qwen training pipeline.

---

# 5.4 Biggest Risks

## Risk 1: Poor Prediction Quality

Hinglish is highly variable and the model may produce irrelevant suggestions.

**Mitigation:**

- Increase training data where possible.
- Improve candidate extraction.
- Use context-aware ranking.
- Compare with n-gram baselines.
- Evaluate Top-1/Top-3/Top-5.
- Investigate internal normalization.

## Risk 2: Subword Tokenization

Qwen predicts subword tokens rather than complete words.

**Mitigation:**

Develop a dedicated whole-word candidate extraction and ranking layer.

## Risk 3: Dataset Sparsity

Some contexts occur rarely or do not occur in the corpus.

**Mitigation:**

Use pretrained contextual representations and investigate additional Hinglish data.

## Risk 4: Inference Latency

Real-time prediction requires fast responses.

**Mitigation:**

- Use the 0.6B model.
- Use MPS locally.
- Cache repeated contexts.
- Limit generated candidates.
- Optimize candidate ranking.
- Investigate quantization.

## Risk 5: Normalization Errors

Incorrect normalization can change the intended meaning.

**Mitigation:**

Use a context-aware normalization model rather than simple dictionary replacement.

---

# 5.5 Project Task Distribution

### Member 1 — Data & Preprocessing

- Dataset collection
- Dataset analysis
- Sentence reconstruction
- Duplicate removal
- Train/validation/test split
- Dataset documentation

### Member 2 — Model & Training

- N-gram baseline
- Qwen model selection
- LoRA configuration
- Fine-tuning
- Training experiments
- Hyperparameter experiments

### Member 3 — Evaluation & Normalization

- Evaluation pipeline
- Top-1/Top-3/Top-5 metrics
- Error analysis
- Hinglish normalization research
- Normalization model development

### Member 4 — Application & Deployment

- FastAPI backend
- Prediction endpoint
- Browser UI
- Suggestion interaction
- Local Mac deployment
- Final demonstration

Responsibilities can be combined if the project has fewer members.

---

# 6. Repository Structure

```text
hinglish-next-word-predictor/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── app.py
│   └── index.html
│
├── training/
│   ├── train.py
│   └── training_config.py
│
├── preprocessing/
│   └── parse_all.py
│
├── evaluation/
│   ├── evaluate.py
│   └── results.md
│
├── model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
│
└── data/
    └── README.md
```

Large raw datasets and the complete pretrained Qwen model should not be unnecessarily committed to GitHub.

---

# 7. Running the Project

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The first execution may download the Qwen3-0.6B base model from Hugging Face.

The trained LoRA adapter is loaded from:

```text
model/
```

---

# 8. Technologies Used

```text
Python
PyTorch
Hugging Face Transformers
PEFT
TRL
FastAPI
HTML
CSS
JavaScript
Qwen3-0.6B
LoRA
Apple MPS
CUDA / NVIDIA Tesla T4
```

---

# 9. Current Status

| Component                         | Status                       |
| --------------------------------- | ---------------------------- |
| Dataset analysis                  | Completed                    |
| Sentence reconstruction           | Completed                    |
| Duplicate removal                 | Completed                    |
| Train/validation/test split       | Completed                    |
| N-gram baseline                   | Completed                    |
| Qwen model selection              | Completed                    |
| LoRA configuration                | Completed                    |
| Qwen LoRA training                | Completed                    |
| Trained adapter saved             | Completed                    |
| Initial evaluation                | Completed                    |
| Word-level evaluation improvement | In progress                  |
| Real-time UI                      | Implemented                  |
| FastAPI inference backend         | Implemented                  |
| Normalization model               | Planned / separate component |
| Final deployment                  | Future work                  |

---

# 10. Normalization

The project investigates Hinglish text normalization as a separate internal component.

The normalization corpus provides:

```text
inputText → normalizedText
```

examples and includes short forms, acronyms, typos, wordplay, Romanized Hindi, splitting, and merging.

The current Qwen next-word model was trained directly on `all.txt`. Therefore, the normalization model has **not been integrated into the current Qwen training pipeline**.

The intended architecture is:

```text
User Input
    ↓
Internal Hinglish Normalization
    ↓
Qwen + LoRA Next-Word Predictor
    ↓
Word Suggestions
```

The normalized text would remain internal and would not need to be displayed to the user.

---

# 11. Conclusion

This project develops a Hinglish-specific next-word prediction system using a pretrained causal language model adapted with LoRA. The system is designed around Romanized Hindi, English-Hindi code mixing, informal spelling, and contextual word usage. A traditional n-gram baseline was investigated to establish the limitations of sparse context-based prediction, while Qwen3-0.6B was selected as the main contextual language model. The resulting model can be integrated into a real-time browser-based writing assistant that provides suggestions while the user types. Future work will focus on rigorous word-level evaluation, improved candidate ranking, context-aware normalization, latency optimization, and deployment as a practical Hinglish writing assistant.

---

# 12. References

1. **A Dataset for Hindi-English Code-Mixed Text Normalization.**  
   Provides the 13,494-pair Hinglish normalization corpus and normalization research basis.

2. **HinGE: A Dataset for Generation and Evaluation of Code-Mixed Hinglish Text.**  
   Investigated as an additional source of Hinglish/code-mixed data.

3. **Qwen3-0.6B-Base.**  
   Pretrained causal language model used for next-word prediction.

---

## Project Goal

The ultimate goal is to build a writing assistant that understands how people actually type Hinglish and provides useful, context-aware next-word suggestions in real time.

```text
Type Hinglish
      ↓
Understand context
      ↓
Predict next word
      ↓
Suggest instantly
```

**Hinglish Next-Word Predictor — a Grammarly-style prediction system for Hinglish.**
