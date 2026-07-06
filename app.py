# app.py — Credit Risk Scoring System
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Risk Scorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS — FIXED for dark mode readability
st.markdown("""
<style>
    /* Main header — WHITE for dark background */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.1rem;
        color: #b0b3b8;
        margin-bottom: 2rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%) !important;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%) !important;
        color: #333 !important;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    /* Sidebar fixes */
    .stSidebar {
        background-color: #1e1e2e;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #ffffff !important;
    }
    .stSidebar label, .stSidebar .stMarkdown {
        color: #e4e6eb !important;
    }
    .stSidebar .stSelectbox label, .stSidebar .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    /* Section headers in main area */
    h3 {
        color: #ffffff !important;
    }
    /* Info box */
    .stInfo {
        background-color: #2d2d44 !important;
        color: #e4e6eb !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cols = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
            'savings_status', 'employment', 'installment_rate', 'personal_status',
            'other_parties', 'residence_since', 'property_magnitude', 'age',
            'other_payment_plans', 'housing', 'existing_credits', 'job',
            'num_dependents', 'own_telephone', 'foreign_worker', 'class']
    df = pd.read_csv(url, sep=' ', names=cols, header=None)
    df['class'] = df['class'].map({1: 0, 2: 1})
    
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    return model, X_train, X_test, y_test, auc, X.columns.tolist()

with st.spinner("Loading model..."):
    model, X_train, X_test, y_test, auc, feature_names = load_and_train()

# HEADER — BIG AND WHITE
st.markdown('<div class="main-header">Credit Risk Scoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable AI for Loan Default Prediction</div>', unsafe_allow_html=True)

st.markdown("---")

# SIDEBAR INPUTS — WHITE TEXT
st.sidebar.markdown("## Applicant Details")
st.sidebar.markdown("Enter the applicant's financial profile below.")
st.sidebar.markdown("---")

checking_status = st.sidebar.selectbox("Checking Account Status", 
    ["A11: < 0 DM", "A12: 0-200 DM", "A13: >= 200 DM", "A14: no checking account"])
duration = st.sidebar.slider("Duration (months)", 4, 72, 12)
credit_history = st.sidebar.selectbox("Credit History",
    ["A30: no credits taken", "A31: all credits paid back", "A32: existing credits paid back",
     "A33: delay in paying off", "A34: critical account"])
credit_amount = st.sidebar.number_input("Credit Amount (DM)", 250, 18424, 5000)
savings_status = st.sidebar.selectbox("Savings Account",
    ["A61: < 100 DM", "A62: 100-500 DM", "A63: 500-1000 DM", "A64: >= 1000 DM", "A65: unknown"])
employment = st.sidebar.selectbox("Employment Duration",
    ["A71: unemployed", "A72: < 1 year", "A73: 1-4 years", "A74: 4-7 years", "A75: >= 7 years"])
installment_rate = st.sidebar.slider("Installment Rate (% of income)", 1, 4, 2)
age = st.sidebar.slider("Age", 19, 75, 35)
property_magnitude = st.sidebar.selectbox("Property",
    ["A121: real estate", "A122: life insurance", "A123: car/other", "A124: no property"])
housing = st.sidebar.selectbox("Housing",
    ["A151: rent", "A152: own", "A153: for free"])
existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 1)
job = st.sidebar.selectbox("Job Type",
    ["A171: unemployed/unskilled", "A172: unskilled resident", "A173: skilled", "A174: highly skilled"])

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Predict Risk", type="primary", use_container_width=True)

# ENCODE INPUTS
checking_map = {"A11: < 0 DM": 0, "A12: 0-200 DM": 1, "A13: >= 200 DM": 2, "A14: no checking account": 3}
credit_hist_map = {"A30: no credits taken": 0, "A31: all credits paid back": 1, 
                   "A32: existing credits paid back": 2, "A33: delay in paying off": 3, "A34: critical account": 4}
savings_map = {"A61: < 100 DM": 0, "A62: 100-500 DM": 1, "A63: 500-1000 DM": 2, "A64: >= 1000 DM": 3, "A65: unknown": 4}
employment_map = {"A71: unemployed": 0, "A72: < 1 year": 1, "A73: 1-4 years": 2, "A74: 4-7 years": 3, "A75: >= 7 years": 4}
property_map = {"A121: real estate": 0, "A122: life insurance": 1, "A123: car/other": 2, "A124: no property": 3}
housing_map = {"A151: rent": 0, "A152: own": 1, "A153: for free": 2}
job_map = {"A171: unemployed/unskilled": 0, "A172: unskilled resident": 1, "A173: skilled": 2, "A174: highly skilled": 3}

input_data = pd.DataFrame([{
    'checking_status': checking_map[checking_status],
    'duration': duration,
    'credit_history': credit_hist_map[credit_history],
    'purpose': 0,
    'credit_amount': credit_amount,
    'savings_status': savings_map[savings_status],
    'employment': employment_map[employment],
    'installment_rate': installment_rate,
    'personal_status': 0,
    'other_parties': 0,
    'residence_since': 2,
    'property_magnitude': property_map[property_magnitude],
    'age': age,
    'other_payment_plans': 0,
    'housing': housing_map[housing],
    'existing_credits': existing_credits,
    'job': job_map[job],
    'num_dependents': 1,
    'own_telephone': 1,
    'foreign_worker': 0,
}])

# PREDICTION
if predict_btn:
    risk_proba = model.predict_proba(input_data)[0][1]
    risk_pct = risk_proba * 100
    
    # Risk level
    if risk_pct > 50:
        risk_class = "risk-high"
        risk_text = "HIGH RISK"
        risk_emoji = "WARNING"
    elif risk_pct > 25:
        risk_class = "risk-medium"
        risk_text = "MEDIUM RISK"
        risk_emoji = "CAUTION"
    else:
        risk_class = "risk-low"
        risk_text = "LOW RISK"
        risk_emoji = "SAFE"
    
    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><div class="metric-value">' + f"{risk_pct:.1f}%" + '</div><div class="metric-label">Default Risk</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="metric-value">' + f"{auc:.3f}" + '</div><div class="metric-label">Model AUC</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card ' + risk_class + '"><div class="metric-value">' + risk_emoji + '</div><div class="metric-label">' + risk_text + '</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SHAP EXPLANATION
    st.markdown("### Why this prediction?")
    st.markdown("*Understanding which factors influenced the risk assessment*")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        contributions = shap_values[0, :, 1]
    elif isinstance(shap_values, list):
        contributions = shap_values[1][0]
    else:
        contributions = shap_values[0]
    
    # Get top 10 features by absolute contribution
    n_show = min(10, len(contributions))
    indices = np.argsort(np.abs(contributions))[::-1][:n_show]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if contributions[i] > 0 else '#27ae60' for i in indices]
    bars = ax.barh(range(n_show), [contributions[i] for i in indices], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i].replace('_', ' ').title() for i in indices], fontsize=11)
    ax.set_xlabel("Impact on Default Risk", fontsize=12, fontweight='bold')
    ax.set_title("Feature Contributions to Risk Prediction", fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, idx) in enumerate(zip(bars, indices)):
        width = bar.get_width()
        label_x = width + 0.001 if width >= 0 else width - 0.001
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{contributions[idx]:+.3f}', 
                ha=ha, va='center', fontsize=9, fontweight='bold',
                color='#e74c3c' if width > 0 else '#27ae60')
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Increases Risk (Red)'),
        Patch(facecolor='#27ae60', label='Decreases Risk (Green)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("""
    **How to read this:** Each bar shows how much a specific feature pushed the prediction 
    toward "High Risk" (red) or "Low Risk" (green). Longer bars = stronger influence.
    """)
    
    st.markdown("---")
    
    # KEY INSIGHTS
    st.markdown("### Key Insights")
    
    # Find top risk factors
    risk_factors = [(feature_names[i], contributions[i]) for i in indices if contributions[i] > 0]
    protective_factors = [(feature_names[i], contributions[i]) for i in indices if contributions[i] < 0]
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RED Risk Factors**")
        for feat, val in risk_factors[:3]:
            st.markdown("- **" + feat.replace('_', ' ').title() + "**: +" + f"{val:.3f}")
    with c2:
        st.markdown("**GREEN Protective Factors**")
        for feat, val in protective_factors[:3]:
            st.markdown("- **" + feat.replace('_', ' ').title() + "**: " + f"{val:.3f}")

else:
    # Default view before prediction
    st.info("Fill in the applicant details in the sidebar and click **Predict Risk** to see the analysis.")
    
    # Show model info
    st.markdown("### About This System")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Dataset**")
        st.markdown("German Credit Data (UCI)")
        st.markdown("1,000 loan applicants")
    with col2:
        st.markdown("**Algorithm**")
        st.markdown("Random Forest Classifier")
        st.markdown("100 estimators")
    with col3:
        st.markdown("**Explainability**")
        st.markdown("SHAP Values")
        st.markdown("Feature-level insights")

# FOOTER
st.markdown("---")
st.caption("Credit Risk Scoring System | Explainable AI for Financial Risk Assessment")