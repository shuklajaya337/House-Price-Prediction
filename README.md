# 🏠 House Price Prediction App

An interactive machine learning web application that predicts California house prices based on property features.

## 🚀 Live Demo
[👉 Click here to open the Live App](YOUR_STREAMLIT_APP_URL_HERE)

## 📸 Features
- 🔮 **Real-time price prediction** using a trained Random Forest model
- 📊 **Model performance dashboard** — Actual vs Predicted, feature importances
- 🔍 **Data insights** — price distributions, correlation heatmap, dataset explorer
- 🎨 Premium dark UI design

## 📦 Tech Stack
| Tool | Purpose |
|------|---------|
| `Streamlit` | Web app framework |
| `scikit-learn` | Random Forest model |
| `pandas` | Data manipulation |
| `matplotlib / seaborn` | Visualizations |

## 📁 Project Structure
```
Project(House)/
├── main.py              # Streamlit web app
├── housing_new.csv      # Dataset (California Housing)
├── requirements.txt     # Python dependencies
└── README.md
```

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run main.py
```

## 📊 Dataset
- **Source**: California Housing Dataset
- **Rows**: 20,640 (20,433 after cleaning)
- **Target**: `median_house_value` ($)
- **Features**: longitude, latitude, housing age, rooms, bedrooms, population, households, income, ocean proximity

## 🤖 Model
- **Algorithm**: Random Forest Regressor (100 trees)
- **Train/Test Split**: 80% / 20%
- **Preprocessing**: StandardScaler + One-Hot Encoding
