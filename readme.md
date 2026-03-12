Below is a **clean GitHub-ready `README.md`** you can **copy–paste directly**.
No edits needed. It already includes your **final results (Accuracy = 0.8079, F1 = 0.8053)**. 🚀

---

# 🌱 Green Patent Classification using Active Learning, QLoRA, and Multi-Agent Systems

This project builds an **AI pipeline to identify climate-change mitigation patents (Y02 classification)** using a combination of:

* Patent embeddings
* Active learning
* Human-in-the-Loop (HITL)
* Domain adaptation with QLoRA
* Multi-Agent debate systems
* Final model fine-tuning

The goal is to **improve classification of green patents** by combining classical machine learning, large language models, and human feedback.

---

# 📊 Dataset

The dataset used in this project contains **1.5 million patent claims** with climate-related technology tags (Y02 classification).

### Relevant Y02 Classes

| Code | Description                 |
| ---- | --------------------------- |
| Y02A | Climate change adaptation   |
| Y02B | Energy efficient buildings  |
| Y02C | Carbon capture & storage    |
| Y02D | ICT energy efficiency       |
| Y02E | Renewable energy            |
| Y02P | Green industrial processes  |
| Y02T | Low-emission transportation |
| Y02W | Waste management            |

---

# 🧠 Project Architecture

The system follows a **four-stage pipeline**:

```
Patent Dataset
      │
      ▼
Baseline Model (PatentSBERTa + Logistic Regression)
      │
      ▼
Active Learning (Uncertainty Sampling)
      │
      ▼
QLoRA Domain Adaptation (Mistral-7B)
      │
      ▼
Multi-Agent Debate System
      │
      ▼
Human-in-the-Loop Review
      │
      ▼
Final PatentSBERTa Fine-Tuning
```

---

# ⚙️ Part A — Baseline Model

### Goal

Train an initial classifier to distinguish **green vs non-green patents**.

### Model

* Embedding model: **PatentSBERTa**
* Classifier: **Logistic Regression**

### Dataset Preparation

A binary label was created:

```
is_green_silver = any(Y02A,Y02B,Y02C,Y02D,Y02E,Y02P,Y02T,Y02W)
```

A balanced dataset was created:

```
25,000 green patents
25,000 non-green patents
```

Total dataset size:

```
50,000 patents
```

---

### Dataset Split

| Dataset          | Percentage |
| ---------------- | ---------- |
| Train (silver)   | 70%        |
| Pool (unlabeled) | 15%        |
| Eval (silver)    | 15%        |

---

### Frozen Embeddings

Patent texts were converted into embeddings using **PatentSBERTa**.

```
X_train_silver_embeddings
X_eval_silver_embeddings
X_pool_unlabeled_embeddings
```

---

### Baseline Classifier

```
LogisticRegression(max_iter=2000)
```

Evaluation metrics used:

* Precision
* Recall
* F1 Score
* Accuracy

Baseline performance was approximately **77% accuracy**.

---

# 🎯 Part B — Active Learning (Uncertainty Sampling)

The goal is to identify **examples where the model is uncertain**.

### Probability of Green Patent

```
p_green = classifier.predict_proba(X)[:,1]
```

### Uncertainty Score

```
u = 1 − 2|p − 0.5|
```

Uncertainty is highest when **p ≈ 0.5**, meaning the model is unsure.

### Selected Data

The **100 most uncertain patents** were selected.

Output file:

```
hitl_green_100.csv
```

These examples are used for **Human-in-the-Loop annotation and LLM reasoning**.

---

# 🤖 Part C — Advanced Architecture

This stage introduces **domain-adapted LLM reasoning and agent debates**.

---

# 1️⃣ Domain Adaptation with QLoRA

Base model used:

```
Mistral-7B-Instruct-v0.2
```

Adapters were trained using:

* **QLoRA**
* 8-bit quantization
* LoRA attention adapters

### LoRA Configuration

```
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
lora_dropout = 0.05
```

### Training Setup

```
batch size = 4
gradient accumulation = 4
max steps = 50
learning rate = 2e-4
```

Output adapter:

```
mistral_qlora_adapter/
```

---

# 🧑‍⚖️ Part C.2 — Multi-Agent Debate System

A **three-agent architecture** evaluates uncertain patents.

### Agents

| Agent    | Role                                        | Model                                       |
| -------- | ------------------------------------------- | ------------------------------------------- |
| Advocate | Argues why the patent is climate-mitigating |Finetunde Qlora                              |
| Skeptic  | Identifies possible greenwashing            |Ollama 2.5-7B instruct                       |
| Judge    | Makes the final classification              |Ollama 2.5-7B instruct                       |

---

### Advocate Agent

The advocate generates an argument explaining **why a patent contributes to climate mitigation**.

Example prompt:

```
Argue why this patent contributes to climate change mitigation.
Focus on renewable energy, efficiency, emissions reduction.
```

Output file:

```
advocate_output.csv
```

---

### Skeptic Agent

The skeptic challenges the advocate by:

* Identifying weak evidence
* Detecting greenwashing
* Questioning environmental claims

---

### Judge Agent

The judge evaluates both arguments and outputs:

```
{
 "label": 0 or 1,
 "confidence": 0-1,
 "rationale": "short explanation"
}
```

Output file:

```
mas_final_output.csv
```

---

# 🧑‍💻 Part D — Human-in-the-Loop (HITL)

Low-confidence predictions are sent to **human reviewers**.

### Threshold

```
confidence < 0.85
```

These samples are exported to:

```
hitl_review.csv
```

Human reviewers assign the final label:

```
is_green_human
```

---

# 🧬 Final Dataset Construction

The final training dataset combines:

| Source        | Description                       |
| ------------- | --------------------------------- |
| Silver labels | Weak labels from original dataset |
| Gold labels   | Human-reviewed labels             |

```
train_final = silver + gold
```

---

# 🧠 Final Model Training

The final model fine-tunes **PatentSBERTa** for sequence classification.

### Training Setup

Optimizer:

```
AdamW
learning rate = 2e-5
```

Batch size:

```
4
```

Epochs:

```
1
```

Final model saved as:

```
patentsberta_green_ft_final/
```

---

# 📈 Final Results

The final model was evaluated on the evaluation dataset.

| Metric       | Score      |
| ------------ | ---------- |
| **Accuracy** | **0.8079** |
| **F1 Score** | **0.8053** |

These results demonstrate that combining:

* Active learning
* Multi-agent reasoning
* Human feedback

can significantly improve **green patent classification**.

---

# 📁 Project Structure

```
project/
│
├── data/
│   ├── train_meta.csv
│   ├── eval_meta.csv
│   ├── pool_meta.csv
│   └── hitl_green_100.csv
│
├── embeddings/
│   ├── X_train_silver_embeddings.npy
│   ├── X_eval_silver_embeddings.npy
│   └── X_pool_unlabeled_embeddings.npy
│
├── mistral_qlora_adapter/
│
├── patentsberta_green_ft_final/
│
├── advocate_output.csv
├── mas_final_output.csv
└── qlora_mas_human.csv
```

---

# 🛠 Technologies Used

| Tool                 | Purpose                   |
| -------------------- | ------------------------- |
| Python               | Core implementation       |
| Hugging Face         | Datasets and Transformers |
| SentenceTransformers | Patent embeddings         |
| PyTorch              | Model training            |
| Scikit-learn         | Baseline classifier       |
| PEFT                 | QLoRA training            |
| CrewAI               | Multi-agent debate system |

---

# 🚀 Key Contributions

✔ Hybrid **ML + LLM + Human feedback pipeline**
✔ **Active learning** for efficient annotation
✔ **Multi-agent reasoning** to detect greenwashing
✔ **QLoRA domain adaptation** for climate patent analysis
✔ Fine-tuned **PatentSBERTa classifier**

---

# 🔮 Future Improvements

* Train on the **full 1.5M patent dataset**
* Increase **human-reviewed gold labels**
* Add **retrieval-augmented reasoning**
* Improve **agent prompts and debate structure**
