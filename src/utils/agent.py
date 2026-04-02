def health_agent_response(query, patient_data, risk_prob):
    """
    Basic rule-based agent (Agent v1)
    """

    query = query.lower()
    bp = patient_data.get("bp")
    chol = patient_data.get("chol")
    glucose = patient_data.get("glucose")
    age = patient_data.get("age")
    response = ""

    if "risk" in query:
        if risk_prob < 0.4:
            response += "Your predicted heart risk is LOW. "
        elif risk_prob < 0.7:
            response += "Your predicted heart risk is MODERATE. "
        else:
            response += "Your predicted heart risk is HIGH. "

    if "blood pressure" in query or "bp" in query:
        if bp > 140:
            response += "Your blood pressure is high, which increases cardiovascular strain. Consider reducing salt intake and stress. "
        elif bp < 90:
            response += "Your blood pressure is low. Monitor for dizziness or fatigue. "
        else:
            response += "Your blood pressure is within a normal range. "

    if "cholesterol" in query:
        if chol > 240:
            response += "High cholesterol detected. This may lead to artery blockage. Reduce saturated fats. "
        elif chol > 200:
            response += "Borderline cholesterol. Lifestyle improvements recommended. "
        else:
            response += "Cholesterol levels are good. "

    if "sugar" in query or "glucose" in query:
        if glucose > 126:
            response += "High blood sugar detected. Risk of diabetes. Consider medical consultation. "
        else:
            response += "Blood sugar appears normal. "

    if response == "":
        response = "I can help explain your health metrics, risk score, or suggest lifestyle improvements."

    return response
    