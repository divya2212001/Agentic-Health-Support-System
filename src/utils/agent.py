import os
from langchain_groq import ChatGroq
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# Part 1 — Tool Definitions


def create_tools(patient_data, risk_prob, vectorstore=None):
    """
    Creates LangChain tools with patient context bound via closure.
    Each tool has access to patient_data, risk_prob, and vectorstore
    without requiring them as tool inputs.
    """

    @tool
    def analyze_risk(query: str) -> str:
        """Use this tool to analyze the patient's heart disease risk probability
        and explain contributing factors. Call this when the user asks about
        their risk level, risk score, or why they are at risk."""

        prob_pct = risk_prob * 100
        level = "LOW" if risk_prob < 0.4 else "MODERATE" if risk_prob < 0.7 else "HIGH"

        factors = []
        bp = patient_data.get('trestbps', 0)
        chol = patient_data.get('chol', 0)
        hr = patient_data.get('thalach', 0)
        fbs = patient_data.get('fbs', 0)
        bmi = patient_data.get('bmi', 0)
        age = patient_data.get('age', 0)
        exang = patient_data.get('exang', 0)
        oldpeak = patient_data.get('oldpeak', 0)
        ca = patient_data.get('ca', 0)

        if bp > 140:
            factors.append(f"Elevated blood pressure ({bp} mmHg — normal < 120)")
        elif bp > 120:
            factors.append(f"Slightly elevated blood pressure ({bp} mmHg — optimal < 120)")
        else:
            factors.append(f"Blood pressure is normal ({bp} mmHg)")

        if chol > 240:
            factors.append(f"High cholesterol ({chol} mg/dL — desirable < 200)")
        elif chol > 200:
            factors.append(f"Borderline high cholesterol ({chol} mg/dL — desirable < 200)")
        else:
            factors.append(f"Cholesterol is within range ({chol} mg/dL)")

        if fbs == 1:
            factors.append("Fasting blood sugar is elevated (> 120 mg/dL)")
        else:
            factors.append("Fasting blood sugar is normal")

        if hr < 100:
            factors.append(f"Max heart rate is low ({hr} BPM)")
        elif hr > 170:
            factors.append(f"Max heart rate is elevated ({hr} BPM)")
        else:
            factors.append(f"Max heart rate is within range ({hr} BPM)")

        if bmi > 30:
            factors.append(f"BMI indicates obesity ({bmi} kg/m²)")
        elif bmi > 25:
            factors.append(f"BMI indicates overweight ({bmi} kg/m²)")

        if age > 60:
            factors.append(f"Age ({age}) is a contributing risk factor")
        elif age > 45:
            factors.append(f"Age ({age}) is a moderate risk factor")

        if exang == 1:
            factors.append("Exercise-induced angina was recorded")

        if oldpeak > 2.0:
            factors.append(f"ST depression is significant ({oldpeak})")

        if ca > 0:
            factors.append(f"Number of major vessels colored by fluoroscopy: {int(ca)}")

        factors_text = "\n".join(f"• {f}" for f in factors)

        return (
            f"RISK ANALYSIS RESULT:\n"
            f"Patient: {patient_data.get('name', 'Unknown')}\n"
            f"Heart Disease Risk Probability: {prob_pct:.1f}%\n"
            f"Risk Category: {level}\n\n"
            f"Contributing Factors:\n{factors_text}"
        )

    @tool
    def interpret_metrics(metric_name: str) -> str:
        """Use this tool to interpret a specific patient vital sign or metric.
        Input should be one of: blood_pressure, cholesterol, heart_rate,
        blood_sugar, bmi, age, or 'all' for a full summary.
        Call this when the user asks about a specific vital or wants to know
        if their values are normal."""

        metric = metric_name.lower().strip()

        interpretations = {
            "blood_pressure": _interpret_bp(patient_data),
            "cholesterol": _interpret_chol(patient_data),
            "heart_rate": _interpret_hr(patient_data),
            "blood_sugar": _interpret_fbs(patient_data),
            "bmi": _interpret_bmi(patient_data),
            "age": _interpret_age(patient_data),
        }

        if metric == "all":
            return "\n\n".join(
                f"--- {k.upper().replace('_', ' ')} ---\n{v}"
                for k, v in interpretations.items()
            )

        # fuzzy match — find the best matching metric
        for key, value in interpretations.items():
            if key in metric or metric in key:
                return value

        return (
            f"Metric '{metric_name}' not recognized. "
            f"Available metrics: {', '.join(interpretations.keys())}, or 'all'."
        )

    @tool
    def search_documents(query: str) -> str:
        """Use this tool to search through uploaded medical documents for
        relevant information. Only use this when the user has uploaded a
        document and asks about its contents, or when you need clinical
        guidelines from the knowledge base."""

        if not vectorstore:
            return "No documents have been uploaded to the knowledge base yet. Ask the user to upload a PDF or TXT file first."

        try:
            results = vectorstore.similarity_search(query, k=3)
            if not results:
                return "No relevant information found in the uploaded documents."

            docs_text = "\n\n".join(
                f"[Document Chunk {i+1}]:\n{doc.page_content}"
                for i, doc in enumerate(results)
            )
            return f"RETRIEVED FROM UPLOADED DOCUMENTS:\n\n{docs_text}"
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    @tool
    def health_recommendations(query: str) -> str:
        """Use this tool to generate personalized health and lifestyle
        recommendations for the patient. Call this when the user asks for
        advice, improvement tips, diet suggestions, or what they should do
        to improve their health."""

        prob_pct = risk_prob * 100
        level = "LOW" if risk_prob < 0.4 else "MODERATE" if risk_prob < 0.7 else "HIGH"

        bp = patient_data.get('trestbps', 0)
        chol = patient_data.get('chol', 0)
        fbs = patient_data.get('fbs', 0)
        bmi = patient_data.get('bmi', 0)

        recs = []

        # Blood pressure recommendations
        if bp > 140:
            recs.append(
                "🩺 BLOOD PRESSURE: Your BP is high. Reduce sodium intake to "
                "< 2,300 mg/day. Consider the DASH diet. Monitor BP twice daily. "
                "Consult your physician about antihypertensive medication."
            )
        elif bp > 120:
            recs.append(
                "🩺 BLOOD PRESSURE: Slightly elevated. Increase physical activity "
                "to 150 min/week of moderate aerobic exercise. Limit alcohol and caffeine."
            )

        # Cholesterol recommendations
        if chol > 240:
            recs.append(
                "🫀 CHOLESTEROL: High levels detected. Reduce saturated fats "
                "(< 6% of daily calories). Add soluble fiber (oats, beans, lentils). "
                "Discuss statin therapy with your doctor."
            )
        elif chol > 200:
            recs.append(
                "🫀 CHOLESTEROL: Borderline high. Increase omega-3 intake "
                "(fish, walnuts, flaxseed). Exercise regularly. Avoid trans fats."
            )

        # Blood sugar recommendations
        if fbs == 1:
            recs.append(
                "🍬 BLOOD SUGAR: Elevated fasting sugar detected. Monitor HbA1c levels. "
                "Reduce refined carbohydrates and sugary drinks. Consider consulting "
                "an endocrinologist."
            )

        # BMI recommendations
        if bmi > 30:
            recs.append(
                "⚖️ WEIGHT: BMI indicates obesity. Target gradual weight loss of "
                "0.5-1 kg/week. Combine diet modification with regular exercise. "
                "Consider consulting a nutritionist."
            )
        elif bmi > 25:
            recs.append(
                "⚖️ WEIGHT: BMI indicates overweight. Maintain a caloric deficit "
                "through balanced meals and regular physical activity."
            )

        # Risk-level-specific advice
        if level == "HIGH":
            recs.append(
                "🚨 HIGH RISK ADVISORY: Schedule an immediate cardiology consultation. "
                "Consider cardiac stress testing. Ensure medication compliance if prescribed. "
                "Avoid strenuous physical activity until cleared by a doctor."
            )
        elif level == "MODERATE":
            recs.append(
                "⚠️ MODERATE RISK ADVISORY: Schedule a check-up within 2 weeks. "
                "Begin lifestyle modifications immediately. Track your vitals daily."
            )
        else:
            recs.append(
                "✅ LOW RISK: Continue current healthy habits. Annual check-ups "
                "recommended. Stay active and maintain balanced nutrition."
            )

        recs.append(
            "⚕️ DISCLAIMER: These are AI-generated suggestions, not medical diagnoses. "
            "Always consult a qualified healthcare provider for medical decisions."
        )

        return (
            f"PERSONALIZED HEALTH RECOMMENDATIONS "
            f"(Risk: {level} — {prob_pct:.1f}%):\n\n"
            + "\n\n".join(recs)
        )

    # Build the final tool list
    tools = [analyze_risk, interpret_metrics, health_recommendations]
    if vectorstore:
        tools.append(search_documents)

    return tools


# ═══════════════════════════════════════════════════════════════
# Metric Interpretation Helpers (used by interpret_metrics tool)
# ═══════════════════════════════════════════════════════════════

def _interpret_bp(data):
    bp = data.get('trestbps', 0)
    if bp < 90:
        category = "Low (Hypotension)"
        detail = "May cause dizziness and fainting. Monitor closely and consult if symptomatic."
    elif bp <= 120:
        category = "Normal"
        detail = "Within healthy range. No action needed."
    elif bp <= 129:
        category = "Elevated"
        detail = "Risk of developing hypertension. Lifestyle changes recommended."
    elif bp <= 139:
        category = "Stage 1 Hypertension"
        detail = "Medication may be needed alongside lifestyle changes."
    else:
        category = "Stage 2 Hypertension"
        detail = "Requires medical attention and likely antihypertensive medication."
    return f"Blood Pressure: {bp} mmHg\nCategory: {category}\nInterpretation: {detail}"


def _interpret_chol(data):
    chol = data.get('chol', 0)
    if chol < 200:
        category = "Desirable"
        detail = "Within healthy range. Maintain current diet."
    elif chol <= 239:
        category = "Borderline High"
        detail = "Monitor closely. Dietary changes and exercise recommended."
    else:
        category = "High"
        detail = "Significantly elevated. Medical intervention may be needed."
    return f"Cholesterol: {chol} mg/dL\nCategory: {category}\nInterpretation: {detail}"


def _interpret_hr(data):
    hr = data.get('thalach', 0)
    if hr < 60:
        category = "Bradycardia"
        detail = "Below normal resting range. May require medical evaluation."
    elif hr <= 100:
        category = "Normal Resting"
        detail = "Within healthy range."
    elif hr <= 170:
        category = "Elevated"
        detail = "May indicate stress, exercise response, or other conditions. Context dependent."
    else:
        category = "Tachycardia"
        detail = "Significantly elevated. Medical evaluation recommended."
    return f"Max Heart Rate: {hr} BPM\nCategory: {category}\nInterpretation: {detail}"


def _interpret_fbs(data):
    fbs = data.get('fbs', 0)
    if fbs == 1:
        status = "Elevated (> 120 mg/dL)"
        detail = "High fasting blood sugar may indicate pre-diabetes or diabetes. Further testing (HbA1c) recommended."
    else:
        status = "Normal (≤ 120 mg/dL)"
        detail = "Within normal fasting range. No immediate concern."
    return f"Fasting Blood Sugar: {status}\nInterpretation: {detail}"


def _interpret_bmi(data):
    bmi = data.get('bmi', 0)
    if bmi < 18.5:
        category = "Underweight"
        detail = "May indicate nutritional deficiency. Consult a dietitian."
    elif bmi < 25:
        category = "Normal"
        detail = "Within healthy range. Maintain current lifestyle."
    elif bmi < 30:
        category = "Overweight"
        detail = "Increased health risks. Lifestyle changes recommended."
    else:
        category = "Obese"
        detail = "Significant health risks. Medical guidance and weight management recommended."
    return f"BMI: {bmi} kg/m²\nCategory: {category}\nInterpretation: {detail}"


def _interpret_age(data):
    age = data.get('age', 0)
    if age < 40:
        risk = "Lower age-related cardiovascular risk. Maintain preventive habits."
    elif age < 55:
        risk = "Moderate age-related risk. Regular screening recommended."
    else:
        risk = "Higher age-related cardiovascular risk. More frequent monitoring advised."
    return f"Age: {age} years\nInterpretation: {risk}"


# ═══════════════════════════════════════════════════════════════
# Part 2 — ReAct Agent + Memory Setup
# ═══════════════════════════════════════════════════════════════

def create_health_agent(patient_data, risk_prob, vectorstore, memory):
    """
    Builds the ReAct agent with tools and memory.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("`GROQ_API_KEY` is not set. Please add it to your `.env` file to activate the AI Agent.")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    tools = create_tools(patient_data, risk_prob, vectorstore)

    patient_context = f"""
Patient Profile:
- Name: {patient_data.get('name', 'Unknown')}
- Age: {patient_data.get('age', 'Unknown')}
- Gender: {'Male' if patient_data.get('sex', 0) == 1.0 else 'Female'}
- BMI: {patient_data.get('bmi', 'Unknown')}

Clinical Vitals:
- Blood Pressure: {patient_data.get('trestbps', 'Unknown')} mmHg
- Cholesterol: {patient_data.get('chol', 'Unknown')} mg/dL
- Max Heart Rate: {patient_data.get('thalach', 'Unknown')} BPM
- Fasting Blood Sugar: {'> 120' if patient_data.get('fbs', 0) == 1.0 else '< 120'} mg/dL

Current Model Assessment:
- Heart Disease Risk Probability: {risk_prob * 100:.1f}%
"""

    system_prefix = f"""You are MediRisk AI, an intelligent clinical assistant.
You have access to the following patient context:
{patient_context}

Answer the user's query professionally, accurately, and compassionately. Try to format your output using markdown for readability.
If the query is unrelated to health or the patient's profile, politely redirect the conversation to health matters.
"""

    prompt = PromptTemplate.from_template(
        system_prefix + """
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        verbose=False, 
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True
    )
    
    return agent_executor


# ═══════════════════════════════════════════════════════════════
# Existing health_agent_response (kept working until Part 2 & 3)
# ═══════════════════════════════════════════════════════════════

def health_agent_response(query, patient_data, risk_prob, vectorstore, memory):
    """
    RAG-powered LLM consulting agent using ReAct and tools.
    Returns (answer, tools_used).
    """
    try:
        agent_executor = create_health_agent(patient_data, risk_prob, vectorstore, memory)
        response = agent_executor.invoke({"input": query})
        
        answer = response.get("output", "I'm sorry, I couldn't generate a response.")
        
        # Extract tools used from intermediate_steps
        tools_used = []
        if "intermediate_steps" in response:
            for action, _ in response["intermediate_steps"]:
                if hasattr(action, "tool"):
                    tools_used.append(action.tool)
                    
        return answer, tools_used
    except Exception as e:
        return f"**Error executing agent:** {str(e)}", []