# RA Flare Prediction System 🏥

An intelligent multimodal machine learning system designed to predict 
Rheumatoid Arthritis (RA) flare risk for the next 3 days using clinical, 
physiological, and environmental data. This system is specifically adapted 
for Sri Lankan patients and tropical climate conditions.

---

## 📌 Project Overview
This research project implements a **Secondary Multisource Metadata Fusion framework** 
where multiple independent models are trained on different datasets and their outputs 
are combined to generate a final prediction.

---

## 🧠 Methodology

### 🔹 Multimodal Data Sources
- Flaredown Autoimmune Symptom Tracker (Kaggle)
- Sleep Health and Lifestyle Dataset (Kaggle)  
- SWELL HRV Dataset (Kaggle)
- Sri Lankan RA Clinical Dataset (PLOS ONE 2022)

### 🔹 Model Architecture
- Individual models trained per modality
- Probability outputs extracted from each model
- Fusion layer using Logistic Regression (meta-classifier)

### 🔹 Hybrid Scoring
- 80% Clinical Score (DAS28/CDAI-inspired)
- 20% Machine Learning Fusion Output

---

## 📊 Model Performance

| Modality | Model | Accuracy |
|---------|------|---------|
| RA Clinical | XGBoost | 94.5% |
| Sleep | Random Forest | 97.3% |
| HRV | Random Forest | 100% |
| Sri Lankan RA | Random Forest | 87.5% |

---

## ⚙️ Features

- Simple patient-friendly questions (no medical knowledge required)
- Real-time weather integration (Sri Lankan cities)
- 3-day flare risk prediction
- Humidity-based risk adjustment (tropical climate)
- Clinical rule-based override system

---

## 🔄 System Workflow

1. User inputs symptoms via console  
2. Weather data is fetched via API  
3. Each modality model generates a probability score  
4. Scores are fused using a meta-classifier  
5. Final flare risk is calculated and displayed  

---

## 🎥 Demo & Presentation

- 🎬 Demo Video:  
  https://drive.google.com/file/d/1b8L1t0CE5Pz0CVhBLDsdnRcjX8JzGxYT/view?usp=drive_link
- 📊 Presentation Slides:  
  https://docs.google.com/presentation/d/1QjLsukGy7AuADmjqd1waOIliCktjqNa-/edit?usp=sharing&ouid=118082880975097147141&rtpof=true&sd=true

---

## ▶️ How to Run

```bash
python 1_train_models.py
python 2_fusion.py
python 3_patient_console.py
