# 🏠 House Price Prediction App

An interactive machine learning web application that predicts California house prices based on property features.
<img src="images/dashboard_mockup.png" alt="Dashboard Screenshot" width="400" />

[![Open Live App](https://img.shields.io/badge/🚀%20Live%20Demo-Click%20to%20Open%20App-8b5cf6?style=for-the-badge)](https://house-price-prediction-4gndjxxfnmnd5nwd4rf8vf.streamlit.app/)

## 🚀 Live Demo
[👉 Click here to open the Live App](https://house-price-prediction-4gndjxxfnmnd5nwd4rf8vf.streamlit.app/)

## 📸 Features
- 🔮 **Real-time price prediction** using a trained Random Forest model
- 📊 **Model performance dashboard** — Actual vs Predicted, feature importances
- 🔍 **Data insights** — price distributions, correlation heatmap, dataset explorer
- 🎨 Premium dark UI design

## 📁 Project Structure
```
House-Price-Prediction/
├── .github/
│   └── workflows/
│       └── ci.yml      # CI/CD test workflow
├── data/
│   └── housing_new.csv # Dataset (California Housing)
├── notebooks/
│   └── HousePricePredictionProject.ipynb # Experimentation notebook
├── tests/
│   └── test_app.py     # Automated unit tests
├── images/
│   └── dashboard_mockup.png # Visualization screenshot
├── .gitignore          # Git ignore file
├── Dockerfile          # Container configuration
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## 📦 Tech Stack
| Tool | Purpose |
|------|---------|
| `Streamlit` | Web app framework |
| `scikit-learn` | Random Forest model |
| `pandas` | Data manipulation |
| `matplotlib / seaborn` | Visualizations |
| `pytest` | Testing framework |
| `docker` | Containerization |

## ▶️ Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app from the root directory
streamlit run src/main.py
```

## 🐳 Docker Deployment
You can build and run this application inside a Docker container:

```bash
# Build the Docker image
docker build -t house-price-predictor .

# Run the container
docker run -p 8501:8501 house-price-predictor
```
Open `http://localhost:8501` in your browser to access the app.

## 🧪 Testing
We use `pytest` for unit testing the model training and loading code:
```bash
# Run all tests
pytest
```

## 🤖 Model Details
- **Algorithm**: Random Forest Regressor (100 trees)
- **Train/Test Split**: 80% / 20%
- **Preprocessing**: StandardScaler + One-Hot Encoding
- **Target**: `median_house_value` ($)
- **Features**: longitude, latitude, housing age, rooms, bedrooms, population, households, income, ocean proximity

## 📥 Input Features & Dataset Description

The application takes 9 property and location features as inputs to predict the house value. The table below details these features, their typical ranges in the dataset, and what they represent:

| Feature Name | Datatype | Range (UI Sliders) | Description |
| :--- | :--- | :--- | :--- |
| **Longitude** | Float | `-124.35` to `-114.31` | Geographic coordinate (east-west coordinate) of the block; higher values are further east. |
| **Latitude** | Float | `32.54` to `41.95` | Geographic coordinate (north-south coordinate) of the block; higher values are further north. |
| **House Age** | Integer | `1` to `52` years | Median age of the houses within the block. |
| **Total Rooms** | Integer | `2` to `39,320` | Total number of rooms in all houses within the block. |
| **Total Bedrooms** | Integer | `1` to `6,445` | Total number of bedrooms in all houses within the block. |
| **Population** | Integer | `3` to `35,682` | Total number of people residing within the block. |
| **Households** | Integer | `1` to `6,082` | Total number of households (groups of people residing in a home unit) in the block. |
| **Median Income** | Float | `0.5` to `15.0` (×$10k) | Median income for households within the block (measured in tens of thousands of USD, e.g., `4.0` represents ~$40,000). |
| **Ocean Proximity** | Categorical | 5 distinct categories | Location relative to the ocean: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, or `NEAR OCEAN`. |

- **Target Variable**: `median_house_value` ($) — Median house value for households within the block.

## 📈 Model Performance & Evaluation Results

The Random Forest Regressor was evaluated on a test set (20% of the dataset) and achieved high prediction accuracy. The key metrics are:

- **Coefficient of Determination ($R^2$ Score)**: **82.62%**
  - *Interpretation*: The model successfully explains approximately 82.62% of the variance in California housing prices.
- **Root Mean Squared Error (RMSE)**: **$48,758**
  - *Interpretation*: On average, the model's price predictions deviate from the actual home values by around $48,758.

### Feature Importances
The Random Forest model assigns weights to features based on how much they reduce variance during training. Below is the relative importance of the key features:

| Rank | Feature | Importance Score | Description / Impact |
| :---: | :--- | :---: | :--- |
| 1 | **Median Income** | `48.50%` | Primary driver of home value; higher income blocks correlate strongly with higher prices. |
| 2 | **Ocean Proximity (INLAND)** | `14.28%` | Strong indicator of lower value; inland homes are significantly cheaper than coastal properties. |
| 3 | **Longitude** | `10.90%` | Geographic impact based on east-west position in California. |
| 4 | **Latitude** | `10.46%` | Geographic impact based on north-south position (e.g. proximity to major cities). |
| 5 | **House Age** | `5.05%` | Older houses can reflect historical value or differences in construction. |
| 6 | **Population** | `3.27%` | Population density within the block. |
| 7 | **Total Rooms** | `2.35%` | Indicator of physical size/scale of houses in the block. |
| 8 | **Total Bedrooms** | `2.23%` | Correlated with rooms, indicating housing capacity. |
| 9 | **Households** | `1.89%` | Number of families/household units within the block. |
| 10 | **Other Ocean categories** | `< 1.00%` | Minor adjustments for Near Bay, Near Ocean, Island, and <1H Ocean. |

