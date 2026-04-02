from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import pandas as pd
import joblib
from pathlib import Path
import datetime

MODEL_PATH = Path("models/logistic_regression.pkl")
OUTPUT_DIR = Path("exports")

def predict_and_export_pdf(patient_data, patient_name, file_name="report"):
    df = pd.DataFrame([patient_data])
    model = joblib.load(MODEL_PATH)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{file_name}.pdf"
    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("<b>HEALTH REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 15))

    info = [
        ["Patient Name", patient_name, "Date", datetime.datetime.now().strftime("%Y-%m-%d")],
        ["Age", str(patient_data["age"]), "Sex", "Male" if patient_data["sex"] == 1 else "Female"]
    ]

    info_table = Table(info, colWidths=[1.5*inch]*4)
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    elements.append(info_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Overview</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    overview = [
        ["Blood Pressure", "Blood Glucose", "Cholesterol"],
        [
            f"{patient_data['trestbps']} mmHg",
            f"{'High' if patient_data['fbs'] == 1 else 'Normal'}",
            f"{patient_data['chol']} mg/dL"
        ]
    ]

    overview_table = Table(overview, colWidths=[2*inch]*3)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 1, colors.grey)
    ]))

    elements.append(overview_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Clinical Details</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    table_data = [["Parameter", "Value"]]
    for k, v in patient_data.items():
        table_data.append([k, str(v)])

    detail_table = Table(table_data, colWidths=[2.5*inch, 2.5*inch])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkgrey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(detail_table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Diagnosis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Result:</b> {result}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Risk Probability:</b> {round(prob, 3)}", styles["Normal"]))
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("_________________________", styles["Normal"]))
    elements.append(Paragraph("Authorized Medical System", styles["Normal"]))

    doc.build(elements)
    return pdf_path