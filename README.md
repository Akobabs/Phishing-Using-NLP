## 🛡️ Phishing Detection System

This project implements an end-to-end phishing detection pipeline powered by a **BERT-based NLP model** and a **Streamlit web interface**. It integrates multiple public datasets, performs data cleaning and balancing, trains a BERT model, and offers a live demo with a heuristic-based predictor. A distraction script is also included to simulate complex backend activity during demos.

---

### 📁 Project Structure

```
data/
├── raw/
│   ├── enron/           # Enron email dataset
│   ├── phishtank/       # PhishTank phishing URLs
│   └── uci.arff         # UCI Phishing Websites dataset
├── intermediate/        # Processed datasets
│   ├── processed_dataset.csv
│   └── balanced_processed_dataset.csv

models/
└── bert_phishing/       # Trained BERT model

results/                 # Training outputs and plots
logs/                    # Training logs

EDA.ipynb                # Dataset analysis and SMOTE balancing
preprocess.py            # Text cleaning and BERT tokenization
train_bert.py            # BERT model training
app.py                   # Streamlit frontend (demo interface)
distract.py              # Fake backend logging script for demos
```

---

### ⚙️ Prerequisites

* Python 3.8+

* Install dependencies:

  ```bash
  pip install torch transformers pandas sklearn imblearn nltk matplotlib seaborn wordcloud streamlit
  ```

* Download NLTK data:

  ```python
  import nltk
  nltk.download('punkt')
  ```

---

### 🚀 Getting Started

#### **Step 1: Data Exploration and Balancing**

Open `EDA.ipynb` to:

* Load and combine Enron, PhishTank, and UCI datasets.
* Filter empty text rows and apply SMOTE (Synthetic Minority Oversampling Technique).
* Generate:

  * `processed_dataset.csv` (imbalanced)
  * `balanced_processed_dataset.csv` (50/50 class balance)

> 📌 **SMOTE Error Fix**: If you encounter a `'SMOTE' object has no attribute 'sample_indices_'` error, replace the logic in Cell 9 with the fixed code provided in the full documentation (above).

#### **Step 2: Preprocessing for BERT**

Run:

```bash
python preprocess.py
```

* Input: `balanced_processed_dataset.csv` or fallback `processed_dataset.csv`
* Output: `preprocessed_dataset.csv` with tokenized fields for BERT

#### **Step 3: Train BERT Model**

Run:

```bash
python train_bert.py
```

* Input: `preprocessed_dataset.csv`
* Output: BERT model saved to `models/bert_phishing`

> 🔧 GPU support: Remove `no_cuda=True` in `TrainingArguments` for GPU acceleration.

#### **Step 4: Launch the Streamlit App**

Run:

```bash
streamlit run app.py
```

Access at [http://localhost:8501](http://localhost:8501)

**App Tabs:**

* **Predict**: One-time email/URL classification.
* **Live Detection**: Real-time simulation of phishing detection.
* **Help**: Usage guide and examples.
* **Sidebar**: History and CSV export of predictions.

> ⚠️ Currently uses a **heuristic-based model** for predictions. BERT integration is explained below.


---

### 📊 Dataset Summary

| Dataset      | Rows   | Phishing % |
| ------------ | ------ | ---------- |
| Enron        | 28,930 | 50.1%      |
| PhishTank    | 64,157 | 100%       |
| UCI          | 5,849  | 51.6%      |
| **Combined** | 57,860 | 75%        |
| **Balanced** | 86,824 | \~50%      |

> Balanced dataset only available if SMOTE runs successfully.

---

### 🛠️ Troubleshooting

* **SMOTE Fails in EDA.ipynb**: Use `processed_dataset.csv` for training (note: may cause class imbalance issues).
* **Slow BERT Training**: Lower `num_train_epochs` or reduce `per_device_train_batch_size`.
* **Streamlit Uses Heuristic Model**: To integrate BERT, modify `app.py`:

```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('models/bert_phishing')
tokenizer = BertTokenizer.from_pretrained('models/bert_phishing')
```

---

### 🔮 Future Enhancements

* ✅ Integrate trained BERT model into `app.py`
* ⚡ Add GPU support for faster training
* 🔍 Expand the heuristic classifier with more rules or hybrid models
* 🌐 Deploy to cloud (e.g., Hugging Face Spaces or Streamlit Sharing)

