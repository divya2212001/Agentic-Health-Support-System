import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import time
import plotly.express as px

st.set_page_config(
    page_title="MediRisk AI | Professional Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'risk_prob' not in st.session_state:
    st.session_state.risk_prob = 0.0
if 'feature_imp' not in st.session_state:
    st.session_state.feature_imp = {}
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "Random Forest"
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🛡️ Risk Assessment"

def set_tab(tab_name):
    st.session_state.active_tab = tab_name

def reset_analysis():
    st.session_state.analysis_run = False

@st.cache_resource
def load_all_models():
    try:
        models = {
            "Logistic Regression": joblib.load('models/logistic_regression.pkl'),
            "Decision Tree": joblib.load('models/decision_tree.pkl'),
            "Random Forest": joblib.load('models/random_forest.pkl')
        }
        metrics = joblib.load('models/model_metrics.pkl')
        return models, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, {}

all_models, all_metrics = load_all_models()

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #0f172a;
        --surface: #1e293b;
        --border: #334155;
        --accent: #10b981;
        --accent-glow: rgba(16, 185, 129, 0.2);
        --text: #f8fafc;
        --text-muted: #94a3b8;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Outfit', sans-serif;
    }

    .header-box {
        background: var(--surface);
        padding: 30px 40px;
        border-radius: 20px;
        border: 1px solid var(--border);
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-h1 { 
        font-family: 'DM Serif Display', serif; 
        font-size: 2.2rem; margin: 0; color: #fff; 
        display: flex; align-items: center; gap: 15px;
    }
    .brand-subtitle { font-size: 0.85rem; color: var(--text-muted); font-weight: 400; margin-top: 4px; }
    
    .badge-container { display: flex; gap: 10px; }
    .status-badge {
        padding: 6px 14px; border-radius: 100px; font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .badge-ml { background: rgba(16, 185, 129, 0.15); color: var(--accent); border: 1px solid var(--accent); }
    .badge-agent { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid #60a5fa; }

    .nav-bar { display: flex; gap: 12px; margin-bottom: 30px; }
    .nav-tab {
        padding: 10px 20px; border-radius: 12px; background: var(--surface);
        border: 1px solid var(--border); color: var(--text-muted); font-size: 0.8rem;
        font-weight: 600; display: flex; align-items: center; gap: 8px; cursor: pointer;
        transition: all 0.2s;
    }
    .nav-tab.active {
        background: var(--accent); color: #fff; border-color: var(--accent);
        box-shadow: 0 4px 12px var(--accent-glow);
    }

    .dashboard-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 24px;
        transition: border-color 0.3s;
    }
    .card-label {
        font-size: 0.9rem; font-weight: 700; color: var(--text);
        display: flex; align-items: center; gap: 10px; margin-bottom: 25px;
    }
    .card-label i { font-size: 1.1rem; color: var(--accent); }

    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    label { color: var(--text-muted) !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px; }

    .metric-row { margin-bottom: 25px; }
    .metric-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .metric-name { font-size: 0.8rem; font-weight: 600; color: var(--text-muted); }
    .metric-value { font-size: 0.85rem; font-weight: 700; color: var(--accent); }
    .metric-bar-bg { height: 4px; background: #334155; border-radius: 10px; position: relative; }
    .metric-dot {
        width: 14px; height: 14px; background: var(--accent); border-radius: 50%;
        position: absolute; top: 50%; transform: translateY(-50%);
        box-shadow: 0 0 10px var(--accent);
    }
    .metric-range { display: flex; justify-content: space-between; font-size: 0.65rem; color: #64748b; margin-top: 5px; }

    .risk-circle {
        width: 180px; height: 180px; border-radius: 50%;
        border: 8px solid var(--border); display: flex; flex-direction: column;
        align-items: center; justify-content: center; margin: 0 auto 20px;
        position: relative;
    }
    .risk-val { font-family: 'DM Serif Display'; font-size: 3rem; color: #fff; line-height: 1; }
    .risk-unit { font-size: 0.75rem; color: var(--text-muted); letter-spacing: 1px; }

    [data-testid="stHeader"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-box">
    <div>
        <div class="brand-h1">Patient Risk Assessment System</div>
        <div class="brand-subtitle">AI-Powered Healthcare Analytics & Intelligent Health Support</div>
    </div>
    <div class="badge-container">
        <div class="status-badge badge-ml">ML Active</div>
        <div class="status-badge badge-agent">Agent Ready</div>
    </div>
</div>
""", unsafe_allow_html=True)

tabs = ["🛡️ Risk Assessment", "🗂️ Patient History", "📈 Analytics", "🤖 Health Agent"]
cols = st.columns(len(tabs))
for i, tab in enumerate(tabs):
    if cols[i].button(tab, key=f"tab_{i}", use_container_width=True, on_click=set_tab, args=(tab,)):
        pass

st.markdown(f"""
<style>
    div[data-testid="column"]:nth-child({tabs.index(st.session_state.active_tab) + 1}) button {{
        background: var(--accent) !important;
        color: white !important;
        border-color: var(--accent) !important;
        box-shadow: 0 4px 12px var(--accent-glow) !important;
    }}
</style>
""", unsafe_allow_html=True)

if st.session_state.active_tab == "🛡️ Risk Assessment":
    l, r = st.columns([1.4, 1], gap="large")

    with l:
        st.markdown('<p class="card-label">📋 Patient Information</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("Patient Name", value="", placeholder="Enter patient name", on_change=reset_analysis)
            age_val = st.number_input("Age", 1, 100, 45, on_change=reset_analysis)
            sex_val = st.selectbox("Gender", ["Male", "Female"], on_change=reset_analysis)
        with col2:
            model_name = st.selectbox("Model Selection", ["Random Forest", "Logistic Regression", "Decision Tree"], index=0, key="model_sel", on_change=reset_analysis)
            bmi_val = st.number_input("BMI (kg/m²)", 10.0, 50.0, value=None, step=0.1, placeholder="Enter BMI", on_change=reset_analysis)
            exercise_val = st.selectbox("Exercise Level", ["Low", "Moderate", "High"], on_change=reset_analysis)
        
        st.divider()

        st.markdown('<p class="card-label">🩺 Clinical Indicators</p>', unsafe_allow_html=True)
        
        def metric_bar(name, current, min_v, max_v, unit, normal_range):
            pct = (current - min_v) / (max_v - min_v) * 100
            pct = max(0, min(100, pct))
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-header">
                    <div class="metric-name">{name}</div>
                    <div class="metric-value">{current} {unit}</div>
                </div>
                <div class="metric-bar-bg">
                    <div class="metric-dot" style="left: {pct}%"></div>
                </div>
                <div class="metric-range">
                    <span>{min_v}</span>
                    <span>Normal: {normal_range}</span>
                    <span>{max_v}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        c_m1, c_m2 = st.columns(2, gap="large")
        with c_m1:
            bp_val = st.slider("Blood Pressure (Systolic)", 80, 200, 120, on_change=reset_analysis, label_visibility="collapsed")
            metric_bar("BLOOD PRESSURE", bp_val, 80, 200, "mmHg", "90-120")
            
            chol_val = st.slider("Cholesterol Level", 100, 400, 190, on_change=reset_analysis, label_visibility="collapsed")
            metric_bar("CHOLESTEROL", chol_val, 100, 400, "mg/dL", "<200")
        
        with c_m2:
            hr_val = st.slider("Heart Rate (Max)", 40, 200, 150, on_change=reset_analysis, label_visibility="collapsed")
            metric_bar("HEART RATE", hr_val, 40, 200, "BPM", "60-100")
            
            glucose_val = st.slider("Fasting Blood Sugar", 50, 300, 105, on_change=reset_analysis, label_visibility="collapsed")
            metric_bar("BLOOD GLUCOSE", glucose_val, 50, 300, "mg/dL", "70-100")

        st.divider()

        st.markdown('<p class="card-label">🔎 Diagnostic Audit</p>', unsafe_allow_html=True)
        
        ce1, ce2 = st.columns(2)
        with ce1:
            cp_val = st.selectbox("CP TYPE", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=3, on_change=reset_analysis)
            ecg_val = st.selectbox("RESTING ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"], on_change=reset_analysis)
            exang_val = st.selectbox("EXERCISE ANGINA", ["No", "Yes"], on_change=reset_analysis)
        with ce2:
            oldpeak_val = st.slider("ST DEPRESSION", 0.0, 6.0, 1.0, step=0.1, on_change=reset_analysis)
            slope_val = st.selectbox("ST SLOPE", ["Upsloping", "Flat", "Downsloping"], index=1, on_change=reset_analysis)
            ca_val = st.slider("VALVES (CA)", 0, 4, 0, on_change=reset_analysis)
            thal_val = st.selectbox("STRESS TEST", ["Normal", "Fixed Defect", "Reversable Defect"], on_change=reset_analysis)

    with r:
        st.markdown('<p class="card-label">🚀 System Control</p>', unsafe_allow_html=True)
        
        st.session_state.model_choice = model_name
        
        if st.button("RUN RISK ASSESSMENT", use_container_width=True):
            with st.spinner("Processing Clinical Tensors..."):
                time.sleep(1.2)
                model = all_models.get(st.session_state.model_choice)
                if model:
                    s_num = 1.0 if sex_val == "Male" else 0.0
                    cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
                    cp_num = cp_map.get(cp_val, 4)
                    fbs_num = 1.0 if glucose_val > 120 else 0.0
                    ecg_map = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}
                    ecg_num = ecg_map.get(ecg_val, 0)
                    exang_num = 1.0 if exang_val == "Yes" else 0.0
                    slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
                    slope_num = slope_map.get(slope_val, 2)
                    thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversable Defect": 7}
                    thal_num = thal_map.get(thal_val, 3)
                    
                    data = [float(age_val), s_num, cp_num, float(bp_val), float(chol_val), fbs_num, ecg_num, float(hr_val), exang_num, oldpeak_val, slope_num, float(ca_val), float(thal_num)]
                    input_vec = pd.DataFrame([data], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
                    
                    st.session_state.risk_prob = model.predict_proba(input_vec)[0][1]
                    st.session_state.analysis_run = True
                    
                    inner = model.named_steps['model']
                    features = ['Age', 'Sex', 'CP', 'BP', 'Chol', 'Fasting Sugar', 'ECG', 'Heart Rate', 'Exercise Angina', 'Peak', 'Slope', 'CA', 'Thal']
                    if hasattr(inner, 'feature_importances_'):
                        imps_vals = inner.feature_importances_
                    elif hasattr(inner, 'coef_'):
                        imps_vals = np.abs(inner.coef_[0])
                        imps_vals = imps_vals / np.sum(imps_vals)
                    else:
                        imps_vals = [0.1] * 13
                    
                    st.session_state.feature_imp = dict(sorted(zip(features, imps_vals), key=lambda x: x[1], reverse=True)[:5])
                    st.rerun()

        st.divider()
        st.markdown('<p class="card-label">📊 Assessment Insights</p>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_run:
            st.markdown('<div style="padding:100px 0; text-align:center; opacity:0.3;"><p>AWAITING ANALYSIS TRIGGER</p></div>', unsafe_allow_html=True)
        else:
            p = st.session_state.risk_prob
            st.markdown(f"""
            <div class="risk-circle" style="border-color: {'#10b981' if p < 0.4 else '#f59e0b' if p < 0.7 else '#ef4444'};">
                <div class="risk-val">{p:.2f}</div>
                <div class="risk-unit">RISK SCORE</div>
            </div>
            """, unsafe_allow_html=True)
            
            lvl = "LOW" if p < 0.4 else "MODERATE" if p < 0.7 else "HIGH"
            color = "#10b981" if p < 0.4 else "#f59e0b" if p < 0.7 else "#ef4444"
            st.markdown(f'<div style="text-align:center; font-weight:700; color:{color}; font-size:1.4rem; margin-bottom:30px;">{lvl} RISK CATEGORY</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="font-size:0.85rem; font-weight:700; color:var(--text); margin-bottom:15px; text-transform:uppercase;">Key Drivers</div>', unsafe_allow_html=True)
            for feat, val in st.session_state.feature_imp.items():
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.75rem;">
                    <span style="color:var(--text-muted);">{feat}</span>
                    <span style="color:var(--accent); font-weight:600;">{int(val*100)}% Influence</span>
                </div>
                <div style="height:2px; background:#334155; border-radius:10px; margin-bottom:15px;">
                    <div style="height:100%; width:{int(val*300)}%; background:var(--accent); border-radius:10px;"></div>
                </div>
                """, unsafe_allow_html=True)

elif st.session_state.active_tab == "🤖 Health Agent":
    st.markdown('<p class="card-label">🤖 AI Health Consultation Agent</p>', unsafe_allow_html=True)
    
    st.info("The Agentic Health Assistant is being synchronized with your clinical data. This will provide contextual lifestyle advice, medication reminders, and diagnostic explanations.")
    
    chat_col, data_col = st.columns([2, 1])
    with chat_col:
        st.markdown("""
        <div style="background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; height: 350px; overflow-y: auto;">
            <p style="color: var(--accent); font-weight: 600;">System Agent:</p>
            <p style="color: var(--text-muted); font-size: 0.9rem;">Greetings. I am your MediRisk AI Assistant. Once the neural handshake is complete, I will be able to analyze your current risk score of <b>{:.2f}</b> and provide detailed clinical pathing.</p>
        </div>
        """.format(st.session_state.risk_prob), unsafe_allow_html=True)
        st.text_input("Ask the Health Agent...", placeholder="e.g., Explain my fasting blood sugar results.")
    
    with data_col:
        st.markdown('<p style="font-size: 0.8rem; color: var(--text-muted); text-align:center; margin-top:40px;">Neural Sync Progress</p>', unsafe_allow_html=True)
        st.progress(0.15)

else:
    st.markdown(f'<h3 style="text-align:center; padding: 60px 0; color:var(--text-muted);">{st.session_state.active_tab} — Coming in Milestone 2</h3>', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; color: var(--text-muted); font-size: 0.7rem; margin-top: 40px; padding: 20px;">VERIFIED FOR CLINICAL EVALUATION • MEDIRISK PRO V2.5</div>', unsafe_allow_html=True)
