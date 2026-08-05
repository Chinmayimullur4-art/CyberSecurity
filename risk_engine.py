"""
risk_engine.py
Custom AI Risk Prioritization Engine — produces a 0-100 score from several
signals (not just CVSS), plus a human-readable explanation of why the
score was assigned.
"""

from datetime import datetime, timezone
from typing import Optional


def _age_days(published_iso: Optional[str]) -> Optional[int]:
    if not published_iso:
        return None
    try:
        published = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - published).days
    except Exception:
        return None


def calculate_risk(enriched: dict) -> dict:
    """
    Weighted risk model:
      - CVSS base score             -> up to 40 pts
      - Active exploitation (KEV)   -> up to 30 pts
      - Known ransomware use        -> up to 10 pts
      - Vulnerability age           -> up to 10 pts
      - Breadth of affected systems -> up to 10 pts
    """
    reasons = []
    score = 0.0

    cvss = enriched.get("cvss_score")
    if isinstance(cvss, (int, float)):
        cvss_pts = round((cvss / 10) * 40, 1)
        score += cvss_pts
        reasons.append(f"CVSS base score {cvss}/10 contributes {cvss_pts} pts")
    else:
        reasons.append("No CVSS score available — treated conservatively")

    if enriched.get("active_exploitation"):
        score += 30
        reasons.append("✓ Actively exploited in the wild (CISA KEV listed) — +30 pts")
    else:
        reasons.append("No confirmed active exploitation on record")

    if enriched.get("kev_ransomware_use") == "Known":
        score += 10
        reasons.append("✓ Known ransomware campaign use — +10 pts")

    age = _age_days(enriched.get("published"))
    if age is not None:
        if age > 365:
            score += 10
            reasons.append(f"✓ Vulnerability is {age} days old and still relevant — +10 pts")
        elif age > 90:
            score += 5
            reasons.append(f"Vulnerability is {age} days old — +5 pts")

    affected = enriched.get("affected_products") or []
    if len(affected) >= 5:
        score += 10
        reasons.append(f"✓ Broad exposure — {len(affected)}+ affected configurations — +10 pts")
    elif affected:
        score += 5
        reasons.append(f"{len(affected)} affected product configuration(s) — +5 pts")

    score = round(min(score, 100), 1)

    if score >= 85:
        priority, color = "Critical", "#FF4D5E"
    elif score >= 65:
        priority, color = "High", "#FFB627"
    elif score >= 35:
        priority, color = "Medium", "#FFD866"
    else:
        priority, color = "Low", "#4ADE80"

    return {"score": score, "priority": priority, "color": color, "reasons": reasons}
