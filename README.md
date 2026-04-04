# RA Flare Prediction System 🏥

A multimodal machine learning system to predict Rheumatoid Arthritis 
flare risk for the next 3 days, designed for Sri Lankan patients.

## Datasets Used
- Flaredown Autoimmune Symptom Tracker (Kaggle)
- Sleep Health and Lifestyle Dataset (Kaggle)  
- SWELL HRV Dataset (Kaggle)
- Sri Lankan RA Clinical Data — Dissanayake et al., PLOS ONE 2022

## Models
- XGBoost (RA Clinical) — 94.5% accuracy
- Random Forest (Sleep) — 97.3% accuracy
- Random Forest (HRV)   — 100% accuracy
- Random Forest (Sri Lankan RA) — 87.5% accuracy
- Meta fusion classifier — Logistic Regression

## Run
```bash
python 1_train_models.py
python 2_fusion.py
python 3_patient_console.py
```

## Features
- Simple plain-language questions for patients
- Live weather by Sri Lankan city
- 3-day flare risk prediction
- Sri Lanka specific humidity warnings
