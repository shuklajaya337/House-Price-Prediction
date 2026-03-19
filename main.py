import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #a78bfa; font-weight: 700; }
    .metric-card p  { margin: 4px 0 0; color: #9ca3af; font-size: 0.9rem; }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(109,40,217,0.4);
        margin: 20px 0;
    }
    .prediction-box h1 { font-size: 3rem; margin: 0; color: white; font-weight: 700; }
    .prediction-box p  { color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-top: 8px; }

    /* Info box */
    .info-box {
        background: rgba(99,102,241,0.15);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #c4b5fd;
    }

    /* Headers */
    h1, h2, h3 { color: #e0e0e0 !important; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #a78bfa;
        border-bottom: 2px solid rgba(167,139,250,0.3);
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* Slider labels */
    label { color: #c4b5fd !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6d28d9, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 20px rgba(109,40,217,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(109,40,217,0.6);
    }

    /* Selectbox / Radio */
    .stSelectbox label, .stRadio label { color: #c4b5fd !important; }
    div[data-baseweb="select"] {
        background: rgba(255,255,255,0.07) !important;
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9ca3af;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(109,40,217,0.25) !important;
        color: #a78bfa !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train Model (cached) ─────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)

@st.cache_data
def load_and_train():
    # Try local file first, then auto-download from public mirror
    # (same dataset & columns as the Kaggle California Housing Prices dataset)
    try:
        data = pd.read_csv("housing_new.csv")
    except FileNotFoundError:
        try:
            import urllib.request
            with st.spinner("📥 Downloading California Housing dataset from public source..."):
                urllib.request.urlretrieve(DATASET_URL, "housing_new.csv")
            data = pd.read_csv("housing_new.csv")
            st.success("✅ Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Could not download dataset: {e}")
            st.stop()

    # Clean
    data.dropna(inplace=True)

    # One-hot encode
    data = pd.get_dummies(data, columns=['ocean_proximity'])

    # Features / Target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    feature_cols = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return model, scaler, feature_cols, X_test, y_test, y_pred, rmse, r2, data


model, scaler, feature_cols, X_test, y_test, y_pred, rmse, r2, data = load_and_train()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("---")
    st.markdown('<div class="info-box">Fill in the house details below and click <strong>Predict Price</strong> to get an instant estimate.</div>', unsafe_allow_html=True)
    st.markdown("### 📍 Location")
    longitude = st.slider("Longitude", -124.35, -114.31, -119.0, 0.01)
    latitude  = st.slider("Latitude",   32.54,   41.95,   35.0,  0.01)

    st.markdown("### 🏘️ Property Details")
    housing_median_age = st.slider("House Age (years)", 1, 52, 20)
    total_rooms        = st.slider("Total Rooms",       2, 39320, 2000)
    total_bedrooms     = st.slider("Total Bedrooms",    1, 6445,  400)
    households         = st.slider("Households",        1, 6082,  400)
    population         = st.slider("Population (area)", 3, 35682, 1200)

    st.markdown("### 💰 Income")
    median_income = st.slider("Median Income (×$10k)", 0.5, 15.0, 4.0, 0.1)

    st.markdown("### 🌊 Ocean Proximity")
    ocean = st.selectbox("Select Location Type", [
        "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"
    ])

    predict_btn = st.button("🔮 Predict Price")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏠 House Price Prediction")
st.markdown("*California Housing Market · Powered by Random Forest*")
st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "🔍 Data Insights"])


# ═══════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{r2*100:.1f}%</h2><p>Model Accuracy (R²)</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h2>${rmse:,.0f}</h2><p>Avg Prediction Error (RMSE)</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h2>{len(data):,}</h2><p>Training Samples</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if predict_btn:
        # Build input row
        input_dict = {col: 0.0 for col in feature_cols}
        input_dict['longitude']          = longitude
        input_dict['latitude']           = latitude
        input_dict['housing_median_age'] = housing_median_age
        input_dict['total_rooms']        = total_rooms
        input_dict['total_bedrooms']     = total_bedrooms
        input_dict['population']         = population
        input_dict['households']         = households
        input_dict['median_income']      = median_income

        ocean_col = f"ocean_proximity_{ocean}"
        if ocean_col in input_dict:
            input_dict[ocean_col] = True

        input_df = pd.DataFrame([input_dict])
        input_sc = scaler.transform(input_df)
        price    = model.predict(input_sc)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <p>🏡 Estimated House Value</p>
            <h1>${price:,.0f}</h1>
            <p>Based on the features you selected</p>
        </div>""", unsafe_allow_html=True)

        # Feature summary
        st.markdown('<div class="section-title">📋 Your Input Summary</div>', unsafe_allow_html=True)
        summary_cols = st.columns(4)
        items = [
            ("📍 Longitude", f"{longitude}"),
            ("📍 Latitude", f"{latitude}"),
            ("🏠 House Age", f"{housing_median_age} yrs"),
            ("🛏️ Bedrooms", f"{total_bedrooms}"),
            ("🚪 Total Rooms", f"{total_rooms}"),
            ("👨‍👩‍👧 Households", f"{households}"),
            ("👥 Population", f"{population}"),
            ("💰 Income", f"${median_income:.1f}×$10k"),
        ]
        for i, (label, val) in enumerate(items):
            with summary_cols[i % 4]:
                st.metric(label, val)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #9ca3af;">
            <div style="font-size: 5rem;">🏠</div>
            <h3 style="color: #a78bfa !important;">Ready to Predict!</h3>
            <p>Adjust the sliders in the sidebar and click <strong style="color:#c4b5fd">Predict Price</strong></p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📈 Actual vs Predicted Prices</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#9ca3af')
        ax.xaxis.label.set_color('#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        ax.title.set_color('#e0e0e0')
        for spine in ax.spines.values():
            spine.set_color('#374151')

    # Scatter plot
    sample = min(3000, len(y_test))
    idx    = np.random.choice(len(y_test), sample, replace=False)
    y_t    = np.array(y_test)[idx]
    y_p    = y_pred[idx]

    axes[0].scatter(y_t, y_p, alpha=0.4, color='#818cf8', s=10)
    axes[0].plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend(facecolor='#1a1a2e', labelcolor='#e0e0e0')

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).nlargest(8)
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(importances)))
    importances.sort_values().plot(kind='barh', ax=axes[1], color=colors)
    axes[1].set_title("Top 8 Feature Importances")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    st.pyplot(fig)

    # Metrics row
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score",  f"{r2:.4f}")
    m2.metric("RMSE",      f"${rmse:,.0f}")
    m3.metric("Test Size", f"{len(y_test):,} rows")
    m4.metric("Model",     "Random Forest")


# ═══════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔍 Dataset Overview</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**House Price Distribution**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#16213e')
        ax2.tick_params(colors='#9ca3af')
        for spine in ax2.spines.values():
            spine.set_color('#374151')
        ax2.hist(data['median_house_value'], bins=50, color='#6366f1', edgecolor='#4338ca', alpha=0.8)
        ax2.set_xlabel("House Price ($)", color='#9ca3af')
        ax2.set_ylabel("Count", color='#9ca3af')
        ax2.title.set_color('#e0e0e0')
        ax2.set_title("Price Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    with col_b:
        st.markdown("**Price by Ocean Proximity**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#1a1a2e')
        ax3.set_facecolor('#16213e')
        ax3.tick_params(colors='#9ca3af')
        for spine in ax3.spines.values():
            spine.set_color('#374151')

        ocean_cols = [c for c in data.columns if 'ocean_proximity_' in c]
        ocean_prices = {}
        for c in ocean_cols:
            label = c.replace('ocean_proximity_', '')
            ocean_prices[label] = data.loc[data[c] == True, 'median_house_value'].mean()

        bars = ax3.bar(ocean_prices.keys(), ocean_prices.values(),
                       color=['#6366f1','#8b5cf6','#a78bfa','#c4b5fd','#818cf8'])
        ax3.set_ylabel("Avg Price ($)", color='#9ca3af')
        ax3.set_title("Avg Price by Location", color='#e0e0e0')
        ax3.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig3)

    # Correlation heatmap
    st.markdown("---")
    st.markdown("**Correlation Heatmap**")
    num_cols = ['longitude','latitude','housing_median_age','total_rooms',
                'total_bedrooms','population','households','median_income','median_house_value']
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.patch.set_facecolor('#1a1a2e')
    ax4.set_facecolor('#1a1a2e')
    sns.heatmap(data[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm',
                ax=ax4, linewidths=0.5, linecolor='#1a1a2e',
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax4.tick_params(colors='#c4b5fd', labelsize=8)
    ax4.set_title("Feature Correlations", color='#e0e0e0', pad=20)
    plt.tight_layout()
    st.pyplot(fig4)

    # Raw data sample
    st.markdown("---")
    st.markdown("**Sample Data (first 10 rows)**")
    st.dataframe(data.head(10), use_container_width=True)
