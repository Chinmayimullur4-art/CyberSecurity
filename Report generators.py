"""
report_generator.py
Generates a professional AI Investigation Report PDF summarising a single
Q&A investigation: question, CVE/risk details, confidence, sources.
"""

from datetime import datetime, timezone
from typing import Optional

from fpdf import FPDF

SEVERITY_COLORS = {
    "Critical": (255, 77, 94),
    "High": (255, 182, 39),
    "Medium": (230, 190, 40),
    "Low": (74, 222, 128),
}


class InvestigationReport(FPDF):
    def header(self):
        self.set_fill_color(13, 19, 28)
        self.rect(0, 0, 210, 22, "F")
        self.set_text_color(47, 230, 199)
        self.set_font("Helvetica", "B", 16)
        self.set_xy(10, 6)
        self.cell(0, 10, "SENTRY // AI Investigation Report", ln=True)
        self.ln(6)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"SENTRY Threat Intel Console -- Page {self.page_no()}", align="C")

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20, 20, 20)
        self.set_fill_color(230, 240, 240)
        self.cell(0, 8, text, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)


def generate_report_pdf(
    question: str,
    answer: str,
    cve_data: Optional[dict],
    risk: Optional[dict],
    confidence: float,
    sources: list,
) -> bytes:
    pdf = InvestigationReport()
    pdf.add_page()

    pdf.section_title("Investigation Summary")
    pdf.body_text(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    pdf.section_title("Question Asked")
    pdf.body_text(question or "N/A")

    if cve_data:
        pdf.section_title(f"Vulnerability Details -- {cve_data.get('cve_id', 'N/A')}")
        pdf.body_text(cve_data.get("description", "N/A"))
        pdf.body_text(
            f"Published: {cve_data.get('published', 'N/A')}\n"
            f"Last Modified: {cve_data.get('last_modified', 'N/A')}\n"
            f"CVSS Score: {cve_data.get('cvss_score', 'N/A')} ({cve_data.get('cvss_severity', 'N/A')})\n"
            f"CWE: {', '.join(cve_data.get('cwe_ids', [])) or 'N/A'}\n"
            f"Active Exploitation (CISA KEV): {'Yes' if cve_data.get('active_exploitation') else 'No'}"
        )
        if cve_data.get("affected_products"):
            pdf.body_text("Affected Products:\n" + "\n".join(cve_data["affected_products"][:10]))
        if cve_data.get("references"):
            pdf.body_text("References:\n" + "\n".join(cve_data["references"][:5]))

    if risk:
        pdf.section_title("Risk Assessment")
        color = SEVERITY_COLORS.get(risk["priority"], (150, 150, 150))
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*color)
        pdf.cell(0, 8, f"Risk Score: {risk['score']}/100   Priority: {risk['priority']}", ln=True)
        pdf.set_text_color(30, 30, 30)
        pdf.body_text("Reasoning:\n" + "\n".join(f"- {r}" for r in risk["reasons"]))

    pdf.section_title("Assistant Answer")
    pdf.body_text(answer or "N/A")

    pdf.section_title("Confidence & Sources")
    pdf.body_text(f"Confidence Score: {confidence:.0f}%")
    if sources:
        pdf.body_text("Sources used:\n" + "\n".join(f"- {s}" for s in sources))

    return bytes(pdf.output())
