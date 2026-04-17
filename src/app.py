import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import time
import plotly.express as px
from utils.agent import health_agent_response
from utils.predict_and_export import predict_and_export_pdf
from langchain.memory import ConversationBufferWindowMemory
st.set_page_config(
    page_title="MediRisk AI | Professional Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = True  
if 'risk_prob' not in st.session_state:
    st.session_state.risk_prob = 0.0
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Risk Assessment"
if 'assessment_done' not in st.session_state:
    st.session_state.assessment_done = False

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

def set_tab(tab_name):
    st.session_state.active_tab = tab_name

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
    .stApp {
        font-family: 'Outfit', sans-serif;
    }

    .header-box {
        background: var(--secondary-background-color);
        padding: 30px 40px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .brand-h1 { 
        font-family: 'DM Serif Display', serif; 
        font-size: 2.2rem; 
        margin: 0; 
        color: var(--text-color); 
        display: flex; 
        align-items: center; 
        gap: 15px;
    }

    .brand-subtitle { 
        font-size: 0.85rem; 
        color: var(--text-color); 
        opacity: 0.7;
        font-weight: 400; 
        margin-top: 4px; 
    }
    
    .badge-container { 
        display: flex; 
        gap: 10px; 
    }

    .status-badge {
        padding: 6px 14px; 
        border-radius: 100px; 
        font-size: 0.65rem; 
        font-weight: 700;
        text-transform: uppercase; 
        letter-spacing: 1px;
    }

    .badge-ml { 
        background: rgba(16, 185, 129, 0.15); 
        color: #10b981; 
        border: 1px solid #10b981; 
    }

    .badge-agent { 
        background: rgba(59, 130, 246, 0.15); 
        color: #3b82f6; 
        border: 1px solid #3b82f6; 
    }

    .card-label {
        font-size: 0.9rem; 
        font-weight: 700; 
        color: var(--text-color);
        display: flex; 
        align-items: center; 
        gap: 10px; 
        margin-bottom: 25px;
    }

    div[data-baseweb="input"], 
    div[data-baseweb="select"] {
        background-color: transparent !important;
        border-radius: 8px !important;
    }

    label { 
        color: var(--text-color) !important; 
        opacity: 0.7 !important;
        font-size: 0.75rem !important; 
        font-weight: 600 !important; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
    }

    .risk-circle {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        border: 8px solid; /* border color inherited */
        background: var(--secondary-background-color);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px;
        position: relative;
        z-index: 10;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .risk-val {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .risk-unit {
        font-size: 0.75rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: -5px;
    }

    /* Metric bar styling from original layout */
    .metric-row { margin-bottom: 20px; }
    .metric-header { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; color: var(--text-color); }
    .metric-name { opacity: 0.8; }
    .metric-value { color: var(--primary-color, #10b981); font-weight: 700; }
    .metric-bar-bg { height: 4px; background: rgba(128,128,128,0.2); border-radius: 4px; position: relative; margin-bottom: 5px; }
    .metric-dot { position: absolute; top: -3px; width: 10px; height: 10px; background: var(--primary-color, #10b981); border-radius: 50%; box-shadow: 0 0 5px var(--primary-color, #10b981); transform: translateX(-50%); }
    .metric-range { display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-color); opacity: 0.6; }

    .dynamic-insight-point {
        padding: 10px 15px;
        border-left: 3px solid var(--primary-color);
        background: var(--secondary-background-color);
        margin-bottom: 10px;
        font-size: 0.9rem;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* CUSTOM TAB NAVIGATION UI */
    button.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 0 !important;
        color: var(--text-color) !important;
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128,128,128,0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    button.stButton > button:hover {
        border-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
    }
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


tabs = ["Risk Assessment", "Patient History", "Analytics", "Health Agent"]
col_nav = st.columns(len(tabs))

for i, tab_name in enumerate(tabs):
    with col_nav[i]:
        st.button(tab_name, key=f"nav_{i}", use_container_width=True, on_click=set_tab, args=(tab_name,))

active_idx = tabs.index(st.session_state.active_tab) + 1
st.markdown(f"""
<style>
    div[data-testid="column"]:nth-child({active_idx}) button.stButton > button {{
        background: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }}
</style>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Risk Assessment Tab
if st.session_state.active_tab == "Risk Assessment":
    l, r = st.columns([1.4, 1], gap="large")

    with l:
        st.markdown('<p class="card-label">Patient Information</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("Patient Name", value="", placeholder="Enter patient name")
            age_val = st.number_input("Age", 1, 100, 1)
            sex_val = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            bmi_val = st.number_input("BMI (kg/m²)", 10.0, 50.0, value=22.0, step=0.1, placeholder="Enter BMI")
            exercise_val = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
        
        st.divider()

        st.markdown('<p class="card-label">Clinical Indicators</p>', unsafe_allow_html=True)
        
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
            bp_val = st.slider("Blood Pressure (Systolic)", 80, 200, 120, label_visibility="collapsed")
            metric_bar("BLOOD PRESSURE", bp_val, 80, 200, "mmHg", "90-120")
            
            chol_val = st.slider("Cholesterol Level", 100, 400, 190, label_visibility="collapsed")
            metric_bar("CHOLESTEROL", chol_val, 100, 400, "mg/dL", "<200")
        
        with c_m2:
            hr_val = st.slider("Heart Rate (Max)", 40, 200, 150, label_visibility="collapsed")
            metric_bar("HEART RATE", hr_val, 40, 200, "BPM", "60-100")
            
            glucose_val = st.slider("Fasting Blood Sugar", 50, 300, 105, label_visibility="collapsed")
            metric_bar("BLOOD GLUCOSE", glucose_val, 50, 300, "mg/dL", "70-100")

        st.divider()

        st.markdown('<p class="card-label">Diagnostic Audit</p>', unsafe_allow_html=True)
        
        ce1, ce2 = st.columns(2)
        with ce1:
            cp_val = st.selectbox("CP TYPE - Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=3)
            ecg_val = st.selectbox("RESTING ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
            exang_val = st.selectbox("EXERCISE ANGINA", ["No", "Yes"])
        with ce2:
            oldpeak_val = st.slider("ST DEPRESSION", 0.0, 6.0, 1.0, step=0.1)
            slope_val = st.selectbox("ST SLOPE", ["Upsloping", "Flat", "Downsloping"], index=1)
            ca_val = st.slider("VALVES (CA)", 0, 4, 0)
            thal_val = st.selectbox("STRESS TEST", ["Normal", "Fixed Defect", "Reversable Defect"])

    # right column - results and insights
    with r:
        st.markdown('<p class="card-label">System Control</p>', unsafe_allow_html=True)
        if st.button("RUN RISK ASSESSMENT", use_container_width=True):
            missing = []

            if not p_name.strip():
                missing.append("Patient Name")

            if age_val <= 0:
                missing.append("Age")

            if bmi_val <= 0:
                missing.append("BMI")

            if missing:
                st.error(
                    "Please fill required fields: "
                    + ", ".join(missing)
                )
                st.session_state.assessment_done = False

            else:
                st.session_state.is_processing = True
                st.session_state.assessment_done = True
            
        st.divider()
        
        st.markdown('<p class="card-label">Assessment Insights</p>', unsafe_allow_html=True)
        
        
        if st.session_state.is_processing:
            with st.spinner("Processing Clinical Tensors..."):
                time.sleep(1.0)

        # RESULT BLOCK - only shows after assessment is run
        model = all_models.get("Random Forest")
        if model and st.session_state.assessment_done:
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
            feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_vec = pd.DataFrame([data], columns=feature_cols)
            
            probability = model.predict_proba(input_vec)[0][1]
            st.session_state.risk_prob = probability

            st.session_state.patient_data = {
                'name': p_name,
                'age': age_val,
                'sex': s_num,
                'cp': cp_num,
                'trestbps': bp_val,
                'chol': chol_val,
                'fbs': fbs_num,
                'restecg': ecg_num,
                'thalach': hr_val,
                'exang': exang_num,
                'oldpeak': oldpeak_val,
                'slope': slope_num,
                'ca': ca_val,
                'thal': thal_num,
                'bmi': bmi_val,
                'predicted_probability': round(probability, 4)
            }

            p = probability
            
            color = "#10b981" if p < 0.4 else "#f59e0b" if p < 0.7 else "#ef4444"
            st.markdown(f"""
            <div class="risk-circle" style="border-color: {color};">
                <div class="risk-val">{p:.2f}</div>
                <div class="risk-unit">HEART HEALTH RISK</div>
            </div>
            """, unsafe_allow_html=True)
            
            lvl = "LOW" if p < 0.4 else "MODERATE" if p < 0.7 else "HIGH"
            st.markdown(f'<div style="text-align:center; font-weight:700; color:{color}; font-size:1.4rem; margin-bottom:20px;">{lvl} RISK CATEGORY</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="font-size:0.85rem; font-weight:700; color:var(--text-color); margin-bottom:15px; text-transform:uppercase;">Clinical Analysis Details</div>', unsafe_allow_html=True)
            
            dynamic_points = []
            if p < 0.4: dynamic_points.append("Overall heart trajectory looks good.")
            elif p < 0.7: dynamic_points.append("Moderate risk profile requires attention.")
            else: dynamic_points.append("High risk profile. Consult a doctor immediately.")

            if bp_val > 140: dynamic_points.append("Elevated Systolic Blood Pressure detected.")
            if chol_val > 240: dynamic_points.append("Cholesterol levels are concerningly high.")
            if glucose_val > 120: dynamic_points.append("High Fasting Blood Sugar. Monitor for pre-diabetes.")
            if bmi_val > 30: dynamic_points.append("BMI indicates obesity.")
            if exang_val == "Yes": dynamic_points.append("Exercise-induced angina recorded.")
            
            if len(dynamic_points) == 1 and p < 0.4:
                dynamic_points.append("All tracked clinical indicators are stable.")

            for point in dynamic_points:
                st.markdown(f'<div class="dynamic-insight-point">{point}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown('<div style="font-size:0.85rem; font-weight:700; color:var(--text-color); margin-bottom:15px; text-transform:uppercase;">Dynamic Key Drivers</div>', unsafe_allow_html=True)
            
            inner = model.named_steps['model']
            features = ['Age', 'Sex', 'CP', 'BP', 'Chol', 'Fasting Sugar', 'ECG', 'Heart Rate', 'Exercise Angina', 'Peak', 'Slope', 'CA', 'Thal']
            
            if hasattr(inner, 'feature_importances_'):
                global_imps = inner.feature_importances_
            elif hasattr(inner, 'coef_'):
                global_imps = np.abs(inner.coef_[0])
                global_imps = global_imps / np.sum(global_imps)
            else:
                global_imps = [0.1] * 13
                
            local_influences = []
            for i, feat in enumerate(features):
                base = global_imps[i]
                multiplier = 1.0
                if feat == 'BP': multiplier += abs(bp_val - 120) / 60.0
                elif feat == 'Chol': multiplier += abs(chol_val - 200) / 100.0
                elif feat == 'Fasting Sugar' and glucose_val > 120: multiplier += 1.5
                elif feat == 'Age': multiplier += abs(age_val - 45) / 40.0
                elif feat == 'Heart Rate': multiplier += abs(hr_val - 150) / 60.0
                elif feat == 'CA': multiplier += (ca_val * 0.4)
                elif feat == 'Peak': multiplier += (oldpeak_val * 0.3)
                
                local_influences.append(base * multiplier)
                
            total_inf = sum(local_influences)
            local_influences = [v/total_inf for v in local_influences]
            
            feature_imp = dict(sorted(zip(features, local_influences), key=lambda x: x[1], reverse=True)[:5])
            
            for feat, val in feature_imp.items():
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.75rem;">
                    <span style="color:var(--text-color); opacity: 0.8;">{feat}</span>
                    <span style="color:var(--primary-color); font-weight:600;">{int(val*100)}% Influence</span>
                </div>
                <div style="height:2px; background:rgba(128,128,128,0.2); border-radius:10px; margin-bottom:15px;">
                    <div style="height:100%; width:{int(val*100)}%; background:var(--primary-color); border-radius:10px;"></div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            
            cl1, cl2 = st.columns(2)
            with cl1:
                if st.button("Save Assessment", use_container_width=True):
                    if not p_name or p_name.strip() == "":
                        st.error("Patient Name required.")
                    else:
                        st.session_state.history.append({
                            "Name": p_name,
                            "Risk Score": round(p, 2),
                            "Risk Level": lvl,
                            "Date": time.strftime("%Y-%m-%d %H:%M")
                        })
                        st.success("Saved.")
            with cl2:
                if st.button("Export PDF", use_container_width=True):
                    if not p_name or p_name.strip() == "":
                        st.error("Patient Name required.")
                    else:
                        try:
                            pdf_path = predict_and_export_pdf(
                                {k: v for k, v in st.session_state.patient_data.items() if k in feature_cols},
                                patient_name=p_name,
                                file_name=f"report_{int(time.time())}"
                            )
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            st.download_button("Download File", data=pdf_bytes, file_name=pdf_path.name, mime="application/pdf")
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")

            # stop loading after rendering results
            st.session_state.is_processing = False

elif st.session_state.active_tab == "Patient History":
    st.markdown("## Patient History")
    st.markdown('<p style="color: var(--text-color); opacity: 0.7; font-size: 0.9rem;">View previously saved assessments.</p>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No patient records yet. Run an assessment and save it first.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

elif st.session_state.active_tab == "Analytics":
    st.markdown("## Analytics Dashboard")
    st.markdown('<p style="color: var(--text-color); opacity: 0.7; font-size: 0.9rem;">System-wide demographic and risk analytics overview.</p>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No data available for analytics.")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown("#### Risk Level Distribution")
            chart_data = df["Risk Level"].value_counts().reset_index()
            chart_data.columns = ["Risk Level", "Count"]
            st.bar_chart(chart_data, x="Risk Level", y="Count", color="Risk Level")
        
        with ca2:
            avg_risk = df["Risk Score"].mean()
            st.metric("Average System Risk Score", f"{avg_risk:.2f}")
            st.markdown("#### Risk Trend")
            df["Date"] = pd.to_datetime(df["Date"])
            df_sorted = df.sort_values("Date")
            st.line_chart(df_sorted.set_index("Date")["Risk Score"])

elif st.session_state.active_tab == "Health Agent":
    st.markdown("## AI Health Consultation Agent")
    st.markdown('<p style="color: var(--text-color); opacity: 0.7; font-size: 0.9rem;">AI Agent powered by LangChain ReAct + Llama 3.1 — equipped with clinical tools, document retrieval, and conversation memory.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; gap: 8px; margin-top: 5px; margin-bottom: 20px;">
        <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: bold;">🧠 MEMORY ACTIVE</span>
        <span style="background: rgba(59, 130, 246, 0.15); color: #3b82f6; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: bold;">🔧 4 TOOLS LOADED</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get('patient_data'):
        st.warning("Please configure your risk assessment first.")
    else:
        from utils.rag import process_document
        
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None

        col_chat, col_rag = st.columns([2, 1], gap="large")
        
        with col_rag:
            st.markdown("### Knowledge Base")
            st.markdown('<p style="font-size: 0.8rem; color: var(--text-color); opacity: 0.7;">Upload clinical guidelines or medical history to give the AI context.</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Doc (.pdf, .txt)", type=["pdf", "txt"], label_visibility="collapsed")
            
            if uploaded_file and st.button("Process Document", use_container_width=True):
                with st.spinner("Embedding document into RAG memory..."):
                    try:
                        vectorstore = process_document(uploaded_file)
                        st.session_state.vectorstore = vectorstore
                        st.success("Document Embedded!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
            if st.session_state.vectorstore:
                st.markdown('<div style="padding: 10px; border-radius: 8px; border: 1px solid #10b981; background: rgba(16, 185, 129, 0.1); color: #10b981; text-align: center; font-weight: bold; font-size: 0.85rem; margin-bottom: 20px;">📄 RAG MEMORY ACTIVE </div>', unsafe_allow_html=True)

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.agent_memory.clear()
                st.rerun()

        with col_chat:
            if len(st.session_state.chat_history) == 0:
                st.info("Hi! Ask me about your health metrics, risk, or upload a document to discuss it.")

            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])
                    if "tools_used" in msg and msg["tools_used"]:
                        with st.expander("🔧 Tools Used"):
                            st.write("\n".join(msg["tools_used"]))

            user_input = st.chat_input("Ask the Health Agent...")

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Agent is reasoning..."):
                        try:
                            answer, tools_used = health_agent_response(
                                user_input,
                                st.session_state.patient_data,
                                st.session_state.risk_prob,
                                st.session_state.vectorstore,
                                st.session_state.agent_memory
                            )
                            st.write(answer)
                            if tools_used:
                                tools_display = ["• " + str(t) for t in tools_used]
                                with st.expander("🔧 Tools Used"):
                                    st.write("\n".join(tools_display))
                                
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": answer,
                                "tools_used": ["• " + str(t) for t in tools_used] if tools_used else None
                            })
                        except Exception as e:
                            st.error(f"Agent error: {e}")

        st.markdown("---")
        st.markdown("**Suggested questions:**")
        st.markdown("- *Analyze my risk factors in detail*")
        st.markdown("- *Are my vitals within normal range?*")
        st.markdown("- *Give me personalized health recommendations*")
        st.markdown("- *What does my uploaded document say about my condition?*")

st.markdown('<div style="text-align: center; color: var(--text-color); opacity: 0.5; font-size: 0.7rem; margin-top: 40px; padding: 20px;">VERIFIED FOR CLINICAL EVALUATION • MEDIRISK PRO V2.5</div>', unsafe_allow_html=True)