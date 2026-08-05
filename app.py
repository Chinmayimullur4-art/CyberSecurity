"""
Cyber Security RAG Assistant
Project 29 — RAG-based threat intelligence Q&A system (Groq edition)

Deploy on Streamlit Community Cloud (share.streamlit.io):
1. Push app.py + requirements.txt to a public GitHub repo.
2. On streamlit.io, "New app" -> point to the repo -> main file: app.py
3. In App settings -> Secrets, add:
   GROQ_API_KEY = "your-key-here"
   (Get a free key at https://console.groq.com/keys)

Note: Groq serves inference only, no embeddings API, so this app embeds
locally with sentence-transformers (all-MiniLM-L6-v2) and uses Groq only
for the final answer generation.
"""

import re
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SENTRY // Threat Intel Console", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Frontend styling only — no backend/session/logic changes below this block
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#0A0E14;
  --panel:#111826;
  --panel-2:#161F2E;
  --border:#232E40;
  --text:#DCE6F0;
  --muted:#7C8B9E;
  --accent:#00E5C7;
  --accent-dim:#0A8F80;
  --danger:#FF4D5E;
  --warn:#FFB627;
  --low:#4ADE80;
}

/* ---------- base canvas ---------- */
[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(180deg, rgba(0,229,199,0.04), transparent 280px),
    repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
    var(--bg);
}
[data-testid="stHeader"]{ background: transparent; }
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color: var(--text); }
h1,h2,h3,h4,h5, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3{
  font-family:'JetBrains Mono', monospace; letter-spacing:0.02em;
}

/* ---------- signature hero: threat scanner ---------- */
.sentry-hero{
  position:relative; overflow:hidden;
  border:1px solid var(--border); border-radius:14px;
  background: radial-gradient(120% 160% at 15% 0%, #101827 0%, #0B111B 60%, #0A0E14 100%);
  padding:28px 32px; margin-bottom:22px;
}
.sentry-hero::before{
  content:""; position:absolute; inset:-50%;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(0,229,199,0.20) 18deg, transparent 40deg, transparent 360deg);
  animation: sweep 4.5s linear infinite;
}
@media (prefers-reduced-motion: reduce){ .sentry-hero::before{ animation:none; } }
@keyframes sweep{ to{ transform:rotate(360deg); } }
.sentry-hero-inner{ position:relative; z-index:1; display:flex; align-items:center; justify-content:space-between; gap:24px; flex-wrap:wrap; }
.sentry-brand{ display:flex; align-items:center; gap:16px; }
.sentry-badge{
  width:52px; height:52px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(145deg, var(--panel-2), var(--panel)); border:1px solid var(--border);
  font-size:26px; box-shadow: 0 0 0 1px rgba(0,229,199,0.08), 0 0 24px rgba(0,229,199,0.15);
}
.sentry-title{ font-family:'JetBrains Mono', monospace; font-size:1.55rem; font-weight:700; margin:0; color:#EEF5FA; }
.sentry-sub{ font-size:0.92rem; color:var(--muted); margin-top:4px; max-width:520px; }
.sentry-status{ display:flex; align-items:center; gap:10px; font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:var(--accent); letter-spacing:0.08em; }
.sentry-dot{ width:9px; height:9px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); animation:pulse 1.8s ease-in-out infinite; }
@keyframes pulse{ 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:0.4; transform:scale(0.8);} }

/* ---------- sidebar console ---------- */
[data-testid="stSidebar"]{
  background: var(--panel); border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3{
  font-size:0.82rem; text-transform:uppercase; letter-spacing:0.12em; color:var(--accent) !important;
}
.console-label{
  font-family:'JetBrains Mono', monospace; font-size:0.75rem; letter-spacing:0.14em;
  color:var(--accent); text-transform:uppercase; margin:2px 0 10px 0; opacity:0.9;
}

/* inputs & buttons */
.stTextInput input, .stFileUploader, textarea{
  background: var(--panel-2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
}
.stButton button, .stFormSubmitButton button{
  background: linear-gradient(145deg, var(--accent-dim), var(--accent)) !important;
  color:#04120F !important; font-weight:700 !important; border:none !important; border-radius:8px !important;
  font-family:'JetBrains Mono', monospace !important; letter-spacing:0.04em;
  transition:transform .15s ease, box-shadow .15s ease;
}
.stButton button:hover{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(0,229,199,0.25); }

/* alerts */
[data-testid="stAlertContentSuccess"], .stSuccess{ border-radius:8px !important; }
div[data-baseweb="notification"]{ border-radius:8px !important; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{
  background: var(--panel) !important; border:1px solid var(--border) !important;
  border-radius:12px !important; padding:4px 6px !important;
}
[data-testid="stChatInput"] textarea{
  background: var(--panel-2) !important; border:1px solid var(--border) !important; color:var(--text) !important;
}

/* ---------- knowledge base pill ---------- */
.kb-pill{
  display:inline-flex; align-items:center; gap:8px;
  background: var(--panel-2); border:1px solid var(--border); border-radius:999px;
  padding:6px 14px; font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:var(--accent);
}
.kb-pill.empty{ color:var(--muted); }

/* ---------- severity chips ---------- */
.sev-chip{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px;
  font-family:'JetBrains Mono', monospace; font-size:0.72rem; font-weight:700; letter-spacing:0.04em; }
.sev-critical{ background:rgba(255,77,94,0.12); color:var(--danger); border:1px solid rgba(255,77,94,0.35); }
.sev-high{ background:rgba(255,182,39,0.12); color:var(--warn); border:1px solid rgba(255,182,39,0.35); }
.sev-medium{ background:rgba(255,182,39,0.10); color:var(--warn); border:1px solid rgba(255,182,39,0.25); }
.sev-low{ background:rgba(74,222,128,0.12); color:var(--low); border:1px solid rgba(74,222,128,0.35); }
.sev-unknown{ background:rgba(124,139,158,0.12); color:var(--muted); border:1px solid rgba(124,139,158,0.35); }

/* ---------- source chip in expander ---------- */
.src-chip{
  display:inline-block; font-family:'JetBrains Mono', monospace; font-size:0.72rem;
  color:var(--accent); background:rgba(0,229,199,0.08); border:1px solid rgba(0,229,199,0.25);
  padding:2px 8px; border-radius:6px; margin-bottom:4px;
}

/* scrollbar */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--border); border-radius:6px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="sentry-hero">
  <div class="sentry-hero-inner">
    <div class="sentry-brand">
      <div class="sentry-badge">🛡️</div>
      <div>
        <p class="sentry-title">SENTRY // Threat Intel Console</p>
        <p class="sentry-sub">Ground your questions in real advisories and CVE data — upload a report or pull a live CVE, then interrogate it.</p>
      </div>
    </div>
    <div class="sentry-status"><span class="sentry-dot"></span> MONITORING</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

CHAT_MODEL = "openai/gpt-oss-120b"  # fast, current Groq model (llama-3.3-70b-versatile is deprecated)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


embedder = load_embedder()

# ---------------------------------------------------------------------------
# API key setup
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
with st.sidebar:
    st.markdown('<p class="console-label">🔑 Access</p>', unsafe_allow_html=True)
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com/keys")
    client = None
    if api_key:
        client = Groq(api_key=api_key)
        st.success("API key configured")
    else:
        st.warning("Enter a Groq API key to enable the assistant.")

    st.divider()
    st.markdown('<p class="console-label">📥 Ingest Intel</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload security advisories / reports (PDF)", type=["pdf"], accept_multiple_files=True
    )

    st.markdown("**Or fetch a live CVE from NVD**")
    cve_id = st.text_input("CVE ID (e.g. CVE-2024-3400)")
    fetch_cve = st.button("📡 Fetch CVE")

    st.divider()
    if st.button("🗑️ Clear knowledge base"):
        st.session_state.pop("index", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("sources", None)
        st.success("Cleared.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def embed_texts(texts):
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return np.array(vectors, dtype="float32")


def embed_query(query: str):
    vector = embedder.encode([query], normalize_embeddings=True)
    return np.array(vector, dtype="float32")


def add_to_index(text: str, source_label: str):
    new_chunks = chunk_text(text)
    if not new_chunks:
        return 0
    vectors = embed_texts(new_chunks)
    dim = vectors.shape[1]
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors
    st.session_state.index.add(vectors)
    st.session_state.chunks.extend(new_chunks)
    st.session_state.sources.extend([source_label] * len(new_chunks))
    return len(new_chunks)


def fetch_cve_from_nvd(cve_id: str):
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.strip().upper()}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None
    cve = vulns[0]["cve"]
    desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
    metrics = cve.get("metrics", {})
    cvss = "N/A"
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics:
            cvss = metrics[key][0]["cvssData"]["baseScore"]
            break
    refs = [r_["url"] for r_ in cve.get("references", [])][:5]
    text = f"CVE ID: {cve_id.upper()}\nCVSS Base Score: {cvss}\nDescription: {desc}\nReferences: {', '.join(refs)}"
    return text, cvss


def severity_badge(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "⚪ Unknown"
    if score >= 9.0:
        return "🔴 Critical"
    if score >= 7.0:
        return "🟠 High"
    if score >= 4.0:
        return "🟡 Medium"
    return "🟢 Low"


def severity_chip_html(score):
    """Presentational-only HTML chip mirroring severity_badge's thresholds. Display use, no logic change."""
    label = severity_badge(score)
    css_class = {
        "🔴 Critical": "sev-critical",
        "🟠 High": "sev-high",
        "🟡 Medium": "sev-medium",
        "🟢 Low": "sev-low",
    }.get(label, "sev-unknown")
    return f'<span class="sev-chip {css_class}">{label} · {score}</span>'


def retrieve(query: str, k=TOP_K):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    qvec = embed_query(query)
    scores, idxs = st.session_state.index.search(qvec, min(k, st.session_state.index.ntotal))
    results = []
    for i in idxs[0]:
        if i == -1:
            continue
        results.append((st.session_state.chunks[i], st.session_state.sources[i]))
    return results


def answer_question(client: Groq, query: str, context_pairs):
    context = "\n\n---\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src in context_pairs)
    prompt = f"""You are a cyber security analyst assistant. Answer the question ONLY using the
context below. If the context doesn't contain the answer, say so clearly rather than guessing.
Cite the source label for each claim you make.

Context:
{context}

Question: {query}

Answer:"""
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Ingest uploads
# ---------------------------------------------------------------------------
if uploaded_files:
    for f in uploaded_files:
        already_added = f.name in st.session_state.sources
        if not already_added:
            with st.spinner(f"Processing {f.name}..."):
                text = extract_pdf_text(f.read())
                n = add_to_index(text, f.name)
            st.sidebar.success(f"Added {n} chunks from {f.name}")

if fetch_cve:
    if not cve_id.strip():
        st.sidebar.error("Enter a CVE ID first.")
    else:
        with st.spinner(f"Fetching {cve_id} from NVD..."):
            try:
                result = fetch_cve_from_nvd(cve_id)
            except Exception as e:
                result = None
                st.sidebar.error(f"NVD lookup failed: {e}")
        if result:
            text, cvss = result
            n = add_to_index(text, cve_id.upper())
            st.sidebar.success(f"Added {cve_id.upper()} — {n} chunk(s) indexed")
            st.sidebar.markdown(severity_chip_html(cvss), unsafe_allow_html=True)
        elif result is None and cve_id.strip():
            st.sidebar.warning("No data found for that CVE ID.")

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.markdown('<p class="console-label" style="margin-top:4px;">💬 Interrogate the intel</p>', unsafe_allow_html=True)

if st.session_state.chunks:
    st.markdown(
        f'<span class="kb-pill">🟢 ONLINE — {len(st.session_state.chunks)} chunks '
        f'· {len(set(st.session_state.sources))} source(s)</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="kb-pill empty">⚪ EMPTY — add a PDF or fetch a CVE from the console to arm the assistant</span>',
        unsafe_allow_html=True,
    )

AVATARS = {"user": "🧑\u200d💻", "assistant": "🛡️"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        st.markdown(msg["content"])

query = st.chat_input("e.g. Is CVE-2024-3400 being actively exploited, and what's the mitigation?")

if query:
    if not client:
        st.error("Please enter a Groq API key in the sidebar first.")
    elif not st.session_state.chunks:
        st.error("Add at least one document or CVE before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(query)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("🔎 Scanning knowledge base and drafting answer..."):
                results = retrieve(query)
                answer = answer_question(client, query, results)
                st.markdown(answer)
                with st.expander("📎 Sources used"):
                    for chunk, src in results:
                        st.markdown(f'<span class="src-chip">{src}</span>', unsafe_allow_html=True)
                        st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})
