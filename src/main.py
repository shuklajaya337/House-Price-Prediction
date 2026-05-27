import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #090514, #120c24, #1a0f30);
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(18, 12, 36, 0.6) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15);
        backdrop-filter: blur(15px);
    }

    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 12px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.2);
    }
    .metric-card h2 { font-size: 1.8rem; margin: 0; color: #a78bfa; font-weight: 700; }
    .metric-card p  { margin: 4px 0 0; color: #94a3b8; font-size: 0.85rem; }

    /* Prediction result box (mockup design with glow effect) */
    .prediction-box {
        background: rgba(109, 40, 217, 0.1);
        border: 2px solid #8b5cf6;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 35px rgba(139, 92, 246, 0.35);
        margin: 20px 0;
        backdrop-filter: blur(15px);
    }
    .prediction-box h1 {
        font-size: 3rem;
        margin: 8px 0;
        color: #ffffff;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.4);
    }
    .prediction-box p  { color: #c4b5fd; font-size: 1rem; margin-top: 8px; }

    /* Info box */
    .info-box {
        background: rgba(139, 92, 246, 0.1);
        border-left: 4px solid #8b5cf6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #d8b4fe;
    }

    /* Headers */
    h1, h2, h3 { color: #f8fafc !important; }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #c4b5fd;
        border-bottom: 2px solid rgba(139, 92, 246, 0.2);
        padding-bottom: 8px;
        margin-bottom: 20px;
        margin-top: 10px;
    }

    /* Slider labels */
    label { color: #c4b5fd !important; }

    /* Buttons (glowing purple style) */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #4f46e5);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.55);
        border-color: rgba(255, 255, 255, 0.25);
    }

    /* Selectbox / Radio */
    .stSelectbox label, .stRadio label { color: #c4b5fd !important; }
    div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 10px 10px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(139, 92, 246, 0.15) !important;
        color: #c4b5fd !important;
        border-bottom: 2px solid #8b5cf6 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Divider */
    hr { border-color: rgba(139, 92, 246, 0.15); }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train Model (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    possible_paths = [
        "data/housing_new.csv",
        "../data/housing_new.csv",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "housing_new.csv"),
        "housing_new.csv"
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        st.error("⚠️ `housing_new.csv` not found. Please ensure it is in the `data/` folder.")
        st.stop()

    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
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
tab1, tab2 = st.tabs(["🔮 Predict & Dashboard", "🔍 Data Insights"])


# ═══════════════════════════════════════════
# TAB 1 — PREDICT & DASHBOARD
# ═══════════════════════════════════════════
with tab1:
    dash_col1, dash_col2 = st.columns([1.1, 1.4])

    with dash_col1:
        # Mini metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h2>{r2*100:.1f}%</h2><p>Accuracy (R²)</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <h2>${rmse:,.0f}</h2><p>Avg Error</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <h2>{len(data):,}</h2><p>Samples</p></div>""", unsafe_allow_html=True)

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
            summary_cols = st.columns(2)
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
                with summary_cols[i % 2]:
                    st.metric(label, val)
        else:
            st.markdown("""
            <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
                <div style="font-size: 5rem;">🏠</div>
                <h3 style="color: #a78bfa !important;">Ready to Predict!</h3>
                <p>Adjust the sliders in the sidebar and click <strong style="color:#c4b5fd">Predict Price</strong> to run the model.</p>
            </div>""", unsafe_allow_html=True)

    with dash_col2:
        st.markdown('<div class="section-title">📈 Model Performance Analysis</div>', unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.patch.set_facecolor('#090514') # Dark theme background matching stApp

        for ax in axes:
            ax.set_facecolor('#120c24') # Dark purple card style
            ax.tick_params(colors='#c4b5fd', labelsize=8)
            ax.xaxis.label.set_color('#c4b5fd')
            ax.yaxis.label.set_color('#c4b5fd')
            ax.title.set_color('#f8fafc')
            for spine in ax.spines.values():
                spine.set_color('#7c3aed') # neon purple border
                spine.set_linewidth(1.0)
            ax.grid(True, color=(1, 1, 1, 0.05), linestyle='--', linewidth=0.5)

        # Scatter plot
        sample = min(3000, len(y_test))
        idx    = np.random.choice(len(y_test), sample, replace=False)
        y_t    = np.array(y_test)[idx]
        y_p    = y_pred[idx]

        axes[0].scatter(y_t, y_p, alpha=0.35, color='#a78bfa', s=8)
        axes[0].plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'r--', lw=1.5, label='Ideal')
        axes[0].set_xlabel("Actual Price ($)", fontsize=9)
        axes[0].set_ylabel("Predicted Price ($)", fontsize=9)
        axes[0].set_title("Actual vs Predicted", fontsize=10, fontweight='bold')
        axes[0].legend(facecolor='#090514', edgecolor='#7c3aed', labelcolor='#ffffff', fontsize=8)

        # Feature importance
        importances = pd.Series(model.feature_importances_, index=feature_cols).nlargest(8)
        colors = plt.cm.Purples(np.linspace(0.4, 0.95, len(importances)))
        importances.sort_values().plot(kind='barh', ax=axes[1], color=colors, edgecolor='#a78bfa', linewidth=0.5)
        axes[1].set_title("Top 8 Feature Importances", fontsize=10, fontweight='bold')
        axes[1].set_xlabel("Importance Score", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # Model Stats Row
        st.markdown("---")
        st.markdown("##### ⚙️ Model Core Specifications")
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.write(f"**R² Score (Test set):** `{r2:.4f}`")
            st.write(f"**Test Set Size:** `{len(y_test):,} rows`")
        with spec_col2:
            st.write(f"**RMSE (Root Mean Square Error):** `${rmse:,.0f}`")
            st.write(f"**Algorithm:** `Random Forest Regressor (100 trees)`")


# ═══════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🔍 Dataset Overview</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**House Price Distribution**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#090514')
        ax2.set_facecolor('#120c24')
        ax2.tick_params(colors='#c4b5fd', labelsize=8)
        ax2.xaxis.label.set_color('#c4b5fd')
        ax2.yaxis.label.set_color('#c4b5fd')
        ax2.title.set_color('#f8fafc')
        for spine in ax2.spines.values():
            spine.set_color('#7c3aed')
            spine.set_linewidth(1.0)
        ax2.hist(data['median_house_value'], bins=50, color='#7c3aed', edgecolor='#a78bfa', alpha=0.8)
        ax2.set_xlabel("House Price ($)", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.set_title("Price Distribution", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)

    with col_b:
        st.markdown("**Price by Ocean Proximity**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#090514')
        ax3.set_facecolor('#120c24')
        ax3.tick_params(colors='#c4b5fd', labelsize=8)
        ax3.yaxis.label.set_color('#c4b5fd')
        ax3.title.set_color('#f8fafc')
        for spine in ax3.spines.values():
            spine.set_color('#7c3aed')
            spine.set_linewidth(1.0)

        ocean_cols = [c for c in data.columns if 'ocean_proximity_' in c]
        ocean_prices = {}
        for c in ocean_cols:
            label = c.replace('ocean_proximity_', '')
            ocean_prices[label] = data.loc[data[c] == True, 'median_house_value'].mean()

        bars = ax3.bar(ocean_prices.keys(), ocean_prices.values(),
                       color=['#7c3aed','#8b5cf6','#a78bfa','#c4b5fd','#818cf8'],
                       edgecolor='#ffffff', linewidth=0.5)
        ax3.set_ylabel("Avg Price ($)", fontsize=9)
        ax3.set_title("Avg Price by Location", fontsize=10, fontweight='bold')
        ax3.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig3)

    # Correlation heatmap
    st.markdown("---")
    st.markdown("**Correlation Heatmap**")
    num_cols = ['longitude','latitude','housing_median_age','total_rooms',
                'total_bedrooms','population','households','median_income','median_house_value']
    fig4, ax4 = plt.subplots(figsize=(10, 5.5))
    fig4.patch.set_facecolor('#090514')
    ax4.set_facecolor('#090514')
    sns.heatmap(data[num_cols].corr(), annot=True, fmt=".2f", cmap='Purples',
                ax=ax4, linewidths=0.5, linecolor='#090514',
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax4.tick_params(colors='#c4b5fd', labelsize=8)
    ax4.set_title("Feature Correlations", color='#f8fafc', pad=20, fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig4)

    # Raw data sample
    st.markdown("---")
    st.markdown("**Sample Data (first 10 rows)**")
    st.dataframe(data.head(10), use_container_width=True)
