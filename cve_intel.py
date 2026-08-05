"""
cve_intel.py
Live threat-intelligence enrichment: NVD (CVE details) + CISA KEV (active
exploitation status). Results are shaped into a single dict that the rest
of the app (risk engine, report generator, chat prompt) can consume.
"""

from typing import Optional

import requests
import streamlit as st

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


@st.cache_data(ttl=3600)
def fetch_kev_catalog() -> dict:
    """Download the CISA Known Exploited Vulnerabilities catalog (cached for
    an hour) and index it by CVE ID for fast lookup."""
    try:
        r = requests.get(KEV_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {v["cveID"].upper(): v for v in data.get("vulnerabilities", [])}
    except Exception:
        return {}


def fetch_nvd_details(cve_id: str) -> Optional[dict]:
    """Fetch full CVE metadata from the NVD API."""
    cve_id = cve_id.strip().upper()
    r = requests.get(NVD_URL.format(cve_id=cve_id), timeout=15)
    r.raise_for_status()
    data = r.json()
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None

    cve = vulns[0]["cve"]
    desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")

    metrics = cve.get("metrics", {})
    cvss_score, cvss_severity, cvss_vector = None, "N/A", None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics:
            m = metrics[key][0]
            cvss_score = m["cvssData"]["baseScore"]
            cvss_severity = m.get("baseSeverity", m["cvssData"].get("baseSeverity", "N/A"))
            cvss_vector = m["cvssData"].get("vectorString")
            break

    cwe_ids = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            if d["lang"] == "en" and d["value"].startswith("CWE"):
                cwe_ids.append(d["value"])

    references = [r_["url"] for r_ in cve.get("references", [])][:8]

    affected = []
    for c in cve.get("configurations", []):
        for node in c.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable"):
                    affected.append(match.get("criteria"))
    affected = list(dict.fromkeys(affected))[:10]

    return {
        "cve_id": cve_id,
        "description": desc,
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "cvss_score": cvss_score,
        "cvss_severity": cvss_severity,
        "cvss_vector": cvss_vector,
        "cwe_ids": cwe_ids,
        "references": references,
        "affected_products": affected,
    }


def enrich_cve(cve_id: str) -> Optional[dict]:
    """Combine NVD details with CISA KEV active-exploitation status into one
    enriched record. Returns None if the CVE is unknown to NVD."""
    details = fetch_nvd_details(cve_id)
    if details is None:
        return None

    kev = fetch_kev_catalog().get(cve_id.strip().upper())
    details["active_exploitation"] = kev is not None
    details["kev_date_added"] = kev.get("dateAdded") if kev else None
    details["kev_due_date"] = kev.get("dueDate") if kev else None
    details["kev_ransomware_use"] = kev.get("knownRansomwareCampaignUse") if kev else None
    details["vendor_advisory"] = kev.get("shortDescription") if kev else None
    details["required_action"] = kev.get("requiredAction") if kev else None
    return details


def enriched_to_text(details: dict) -> str:
    """Flatten an enriched CVE record into plain text so it can be chunked
    and stored in the vector index alongside uploaded documents."""
    lines = [
        f"CVE ID: {details['cve_id']}",
        f"Description: {details.get('description', 'N/A')}",
        f"Published: {details.get('published', 'N/A')}",
        f"Last Modified: {details.get('last_modified', 'N/A')}",
        f"CVSS Score: {details.get('cvss_score', 'N/A')} ({details.get('cvss_severity', 'N/A')})",
        f"CWE: {', '.join(details.get('cwe_ids', [])) or 'N/A'}",
        f"Active Exploitation (CISA KEV): {'Yes' if details.get('active_exploitation') else 'No'}",
    ]
    if details.get("required_action"):
        lines.append(f"CISA Required Action: {details['required_action']}")
    if details.get("affected_products"):
        lines.append("Affected Products: " + "; ".join(details["affected_products"]))
    if details.get("references"):
        lines.append("References: " + ", ".join(details["references"]))
    return "\n".join(lines)
