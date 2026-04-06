from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import pandas as pd
import joblib
from pathlib import Path
import datetime

MODEL_PATH = Path("models/random_forest.pkl")
OUTPUT_DIR = Path("exports")

def predict_and_export_pdf(patient_data, patient_name, file_name="report"):
    df = pd.DataFrame([patient_data])
    model = joblib.load(MODEL_PATH) 
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    try:
        sex_value = int(float(patient_data.get("sex", 0)))
        sex_label = "Male" if sex_value == 1 else "Female"
    except:
        sex_label = "Unknown"

    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{file_name}.pdf"
    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("<b>MEDICAL REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 15))
    info = [
        ["Patient Name", patient_name],
        ["Age", f"{patient_data['age']} years"],
        ["Sex", sex_label],
        ["Date", datetime.datetime.now().strftime("%Y-%m-%d")]
    ]
    info_table = Table(info, colWidths=[2.5*inch, 3.5*inch])
    info_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Clinical Details</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    vitals = [
        ["Parameter", "Value"],
        ["Blood Pressure", f"{patient_data['trestbps']} mmHg"],
        ["Cholesterol", f"{patient_data['chol']} mg/dL"],
        ["Blood Sugar", "High" if int(float(patient_data['fbs'])) == 1 else "Normal"],
        ["Max Heart Rate", f"{patient_data['thalach']} bpm"]
    ]

    vitals_table = Table(vitals, colWidths=[3*inch, 3*inch])
    vitals_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.grey)
    ]))

    elements.append(vitals_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Diagnosis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    color = "red" if pred == 1 else "green"
    elements.append(Paragraph(
        f"<b>Result:</b> <font color='{color}'>{result}</font>",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"<b>Risk Probability:</b> {round(prob, 3)}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("_________________________", styles["Normal"]))
    elements.append(Paragraph("Authorized Medical System", styles["Normal"]))
    doc.build(elements)
    return pdf_path