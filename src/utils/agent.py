def health_agent_response(query, patient_data, risk_prob):
    """
    Improved rule-based agent (Agent v2)
    """

    query = query.lower()
    response = ""

    bp = patient_data.get("bp", 120)
    chol = patient_data.get("chol", 180)
    glucose = patient_data.get("glucose", 100)
    age = patient_data.get("age", 30)

    if "risk" in query or "why" in query or "condition" in query:
        if risk_prob < 0.4:
            response += "Your predicted heart risk is LOW. "
        elif risk_prob < 0.7:
            response += "Your predicted heart risk is MODERATE. "
        else:
            response += "Your predicted heart risk is HIGH. "

        if bp < 130:
            response += "Your blood pressure is within a healthy range. "
        else:
            response += "Elevated blood pressure increases your risk. "

        if chol < 200:
            response += "Cholesterol levels are controlled. "
        else:
            response += "High cholesterol may lead to artery blockage. "

        if glucose < 120:
            response += "Blood sugar is normal. "
        else:
            response += "High glucose can increase cardiovascular risk. "

    if "blood pressure" in query or "bp" in query:
        if bp > 140:
            response += "Your blood pressure is high. Reduce salt intake and manage stress. "
        elif bp < 90:
            response += "Your blood pressure is low. Monitor for dizziness. "
        else:
            response += "Your blood pressure is within a normal range. "

    if "cholesterol" in query:
        if chol > 240:
            response += "High cholesterol detected. Reduce saturated fats. "
        elif chol > 200:
            response += "Borderline cholesterol. Lifestyle improvements recommended. "
        else:
            response += "Cholesterol levels are good. "

    if "sugar" in query or "glucose" in query:
        if glucose > 126:
            response += "High blood sugar detected. Consider medical consultation. "
        else:
            response += "Blood sugar appears normal. "

    if "improve" in query or "health" in query:
        response += """
To improve your heart health:
- Exercise regularly (30 mins daily)
- Maintain a balanced diet
- Reduce salt and sugar intake
- Monitor vitals regularly
"""

    if response == "":
        response = f"""
Your current risk score is {round(risk_prob, 2)}.

You can ask about:
- Your risk level
- Blood pressure
- Cholesterol
- How to improve health
"""

    response += "\n\nTip: Stay active and maintain a healthy lifestyle."

    return response