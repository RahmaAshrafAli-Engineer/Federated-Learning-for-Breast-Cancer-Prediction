
# 🧬 FedCare AI
### Privacy-Aware Breast Cancer Prediction Using Federated Learning

<p align="center">
  <strong>Collaborative AI. Decentralized Data. Smarter Predictions.</strong>
</p>

---

## 📌 Overview

**FedCare AI** is an academic machine learning project that explores how Federated Learning can enable multiple healthcare institutions to collaboratively train an AI model without pooling their local datasets into one centralized training dataset.

The project simulates three hospitals, each with its own local breast cancer dataset. Using **Federated Averaging (FedAvg)**, the local model updates are aggregated to build a global model that can benefit from knowledge learned across participating clients.

A Streamlit dashboard provides an interactive interface for exploring model performance, comparing evaluation results, and generating predictions.

> **Project scope:** This is an educational prototype using a simulated multi-hospital environment. It is not a clinically validated medical system.

## 🎯 Objectives

- Explore Federated Learning for collaborative healthcare AI.
- Simulate distributed training across three hospital clients.
- Implement the FedAvg aggregation workflow.
- Evaluate the global model using classification metrics.
- Build an interactive dashboard for model evaluation and prediction.
- Demonstrate how collaborative training can work without directly combining local training datasets.

## 🏥 How It Works

```text
 Hospital 1          Hospital 2          Hospital 3
 Local Data          Local Data          Local Data
     |                   |                   |
 Local Training      Local Training      Local Training
     |                   |                   |
     +-------------------+-------------------+
                         |
                  Model Updates
                         |
                  Federated Server
                         |
                   FedAvg Aggregation
                         |
                    Global Model
                         |
                  Model Evaluation
                         |
                  Streamlit Dashboard
```

Each hospital trains a local model using its own data. The federated server aggregates the participating model updates using FedAvg and distributes the updated global model for subsequent rounds.

**Privacy note:** Federated Learning reduces the need to transfer raw training data, but model updates can still leak information. This prototype should not be considered a complete privacy or security solution.

## 🧠 Machine Learning Approach

| Component | Description |
|---|---|
| Learning paradigm | Federated Learning |
| Aggregation algorithm | Federated Averaging (FedAvg) |
| Client simulation | Three hospital clients |
| Application domain | Breast cancer classification |
| Evaluation | Accuracy, Recall, and ROC-AUC |
| Interface | Streamlit dashboard |

## 📊 Model Evaluation

The following results were observed in the current experimental run.

| Model / Client | Accuracy | Recall |
|---|---:|---:|
| Hospital 1 local model | 95.6% | 97.6% |
| Hospital 2 local model | 93.0% | 81.0% |
| Hospital 3 local model | 89.5% | 88.1% |
| Centralized reference model | 98.2% | Not reported |
| Federated global model | 97.4% | 95.2% |

**Federated model ROC-AUC:** 0.997

These values describe one experimental run and may vary with the data split, preprocessing, model configuration, and random seed. The centralized reference is included as a comparison baseline; the table alone does not establish that federated training is superior.

## 🛠️ Tech Stack

- **Python** — Core programming language
- **Pandas & NumPy** — Data handling and numerical computation
- **Scikit-learn** — Machine learning and evaluation
- **Joblib** — Model serialization
- **Plotly** — Interactive visualizations
- **Streamlit** — Interactive web dashboard

## 📁 Project Structure

```text
FedCare-AI/
├── 1_prepare_data.py
├── 2_train_federated (1).py
├── 3_app.py
├── hospital_1.csv
├── hospital_2.csv
├── hospital_3.csv
├── global_test.csv
├── federated_model.pkl
├── results.json
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FedCare-AI.git
cd FedCare-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Use compatible Python and library versions for the saved model artifact.

### 4. Run the dashboard

```bash
python -m streamlit run "3_app.py"
```

The dashboard should open in your browser.

## 🖥️ Dashboard Features

- **Overview:** Project introduction and key evaluation metrics.
- **AI Prediction:** Interactive predictions based on the model's expected input features.
- **Federated Training:** Exploration of federated learning results.
- **Model Evaluation:** Classification metrics and performance visualizations.

## 🔐 Privacy and Responsible Use

This project demonstrates a federated training workflow, not a production-ready healthcare deployment.

- Do not upload real patient records or personally identifiable medical information.
- Federated Learning alone does not guarantee privacy.
- Clinical use requires independent validation, appropriate security controls, expert oversight, and regulatory review.
- Predictions must not be used as a substitute for professional medical advice or diagnosis.

## 🔬 Future Improvements

- Integrate the Flower framework for a more realistic federated training setup.
- Evaluate non-IID data distributions across hospitals.
- Compare FedAvg with FedProx.
- Add stronger privacy mechanisms, such as differential privacy.
- Evaluate robustness and generalization across clients.
- Improve experiment tracking and reproducibility.
- Validate the system on appropriately governed and clinically relevant datasets.

## 👩‍💻 Author

**Rahma Ashraf Ali Faragallah**  
AI Lead | Machine Learning | Deep Learning | Federated Learning

## ⚠️ Disclaimer

FedCare AI is an academic prototype developed for educational and research purposes. It is not intended for clinical diagnosis, treatment decisions, or real-world medical deployment.

---

<p align="center">
  <strong>FedCare AI — Exploring Collaborative Intelligence for Healthcare</strong>
</p>
