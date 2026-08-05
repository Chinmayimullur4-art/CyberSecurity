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

Note on requirements: st.chat_input's built-in file attachment (accept_file)
requires Streamlit >= 1.43.0 — requirements.txt has been bumped accordingly.
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

CHAT_MODEL = "openai/gpt-oss-120b"  # fast, current Groq model (llama-3.3-70b-versatile is deprecated)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


embedder = load_embedder()

# ---------------------------------------------------------------------------
# Session state (initialized early so the UI shell can read it on first paint)
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
# Frontend styling only — everything below is presentation, not logic
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#05070B;
  --panel:#0D131C;
  --panel-2:#131B27;
  --border:#1F2B3B;
  --text:#E7EEF5;
  --muted:#7688A0;
  --accent:#2FE6C7;
  --accent-dim:#0F8F7C;
  --accent2:#7B61FF;
  --danger:#FF4D5E;
  --warn:#FFB627;
  --low:#4ADE80;
}

/* ---------- base canvas ---------- */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1px 1px at 20px 30px, rgba(255,255,255,0.06) 100%, transparent),
    radial-gradient(1px 1px at 120px 80px, rgba(255,255,255,0.05) 100%, transparent),
    radial-gradient(1px 1px at 60px 160px, rgba(255,255,255,0.05) 100%, transparent),
    radial-gradient(1px 1px at 200px 220px, rgba(255,255,255,0.04) 100%, transparent),
    radial-gradient(160% 120% at 15% -10%, rgba(123,97,255,0.07), transparent 55%),
    radial-gradient(120% 100% at 90% 0%, rgba(47,230,199,0.06), transparent 50%),
    var(--bg);
  background-size: 260px 260px, 260px 260px, 260px 260px, 260px 260px, auto, auto, auto;
}
[data-testid="stHeader"]{ background: transparent; }
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color: var(--text); }
h1,h2,h3,h4,h5{ font-family:'Space Grotesk', sans-serif; }

/* ---------- signature: the Sentinel + hero ---------- */
.sentry-hero{
  position:relative; overflow:hidden;
  border:1px solid var(--border); border-radius:18px;
  background: linear-gradient(180deg, #0B1119 0%, #070B11 100%);
  padding:34px 38px; margin-bottom:26px;
}
.sentry-hero::before{
  content:""; position:absolute; inset:-60%;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(47,230,199,0.14) 16deg, transparent 34deg, transparent 360deg);
  animation: sweep 5s linear infinite;
}
@media (prefers-reduced-motion: reduce){ .sentry-hero::before, .sentinel-ring, .sentinel-core{ animation:none !important; } }
@keyframes sweep{ to{ transform:rotate(360deg); } }
.sentry-hero-inner{ position:relative; z-index:1; display:flex; align-items:center; justify-content:space-between; gap:28px; flex-wrap:wrap; }
.sentry-brand{ display:flex; align-items:center; gap:22px; }
.sentry-title{ font-family:'Space Grotesk', sans-serif; font-size:2rem; font-weight:700; margin:0; color:#F3F8FC; letter-spacing:-0.01em; }
.sentry-sub{ font-size:0.96rem; color:var(--muted); margin-top:6px; max-width:480px; line-height:1.5; }
.sentry-status{ display:flex; align-items:center; gap:10px; font-family:'JetBrains Mono', monospace; font-size:0.76rem; letter-spacing:0.1em; padding:8px 14px; border-radius:999px; border:1px solid var(--border); background:rgba(255,255,255,0.02); }
.sentry-status.armed{ color:var(--accent); }
.sentry-status.standby{ color:var(--muted); }
.sentry-dot{ width:8px; height:8px; border-radius:50%; background:currentColor; box-shadow:0 0 10px currentColor; animation:pulse 1.8s ease-in-out infinite; }
@keyframes pulse{ 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:0.35; transform:scale(0.75);} }

/* Sentinel character: a watching orb/eye */
.sentinel-wrap{ position:relative; width:74px; height:74px; flex:none; }
.sentinel-ring{ position:absolute; inset:0; border-radius:50%; border:1.5px dashed rgba(47,230,199,0.35); animation: spin 7s linear infinite; }
.sentinel-ring.fast{ animation-duration:1.6s; border-color:rgba(123,97,255,0.55); }
@keyframes spin{ to{ transform:rotate(360deg); } }
.sentinel-core{
  position:absolute; inset:12px; border-radius:50%;
  background: radial-gradient(circle at 35% 30%, #DFFFFA 0%, var(--accent) 28%, var(--accent2) 75%, #241B3D 100%);
  box-shadow: 0 0 22px rgba(47,230,199,0.45), 0 0 46px rgba(123,97,255,0.25);
  animation: breathe 2.4s ease-in-out infinite;
}
.sentinel-core.dim{ filter: grayscale(0.55) brightness(0.6); box-shadow:0 0 12px rgba(47,230,199,0.15); }
.sentinel-core.think{ animation: breathe 0.7s ease-in-out infinite; }
@keyframes breathe{ 0%,100%{ transform:scale(1);} 50%{ transform:scale(0.86);} }
.sentinel-pupil{ position:absolute; inset:26px; border-radius:50%; background:#05070B; }

.thinking-strip{ display:flex; align-items:center; gap:12px; font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:var(--accent2); margin:6px 0 2px 0; }

/* ---------- section labels ---------- */
.console-label{
  font-family:'JetBrains Mono', monospace; font-size:0.74rem; letter-spacing:0.14em;
  color:var(--accent); text-transform:uppercase; margin:18px 0 10px 0; opacity:0.92;
}
[data-testid="stSidebar"] .console-label{ margin-top:2px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background: var(--panel); border-right:1px solid var(--border); }
.mission-stat{ display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:var(--muted); padding:5px 0; border-bottom:1px dashed var(--border); }
.mission-stat b{ color:var(--text); }

/* inputs & buttons */
.stTextInput input, textarea{
  background: var(--panel-2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
}
.stButton button, .stFormSubmitButton button{
  background: linear-gradient(145deg, var(--accent-dim), var(--accent)) !important;
  color:#03110D !important; font-weight:700 !important; border:none !important; border-radius:8px !important;
  font-family:'JetBrains Mono', monospace !important; letter-spacing:0.03em;
  transition:transform .15s ease, box-shadow .15s ease;
}
.stButton button:hover{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(47,230,199,0.22); }

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{
  background: var(--panel) !important; border:1px solid var(--border) !important;
  border-radius:12px !important; padding:4px 6px !important;
}
[data-testid="stChatInput"] textarea{
  background: var(--panel-2) !important; border:1px solid var(--border) !important; color:var(--text) !important;
}
[data-testid="stChatInput"]{
  border:1px solid var(--border) !important; border-radius:16px !important; background: var(--panel) !important;
}
[data-testid="stChatInput"] button{
  color: var(--accent) !important;
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

/* ---------- source / attachment chip ---------- */
.src-chip{
  display:inline-block; font-family:'JetBrains Mono', monospace; font-size:0.72rem;
  color:var(--accent); background:rgba(47,230,199,0.08); border:1px solid rgba(47,230,199,0.25);
  padding:2px 8px; border-radius:6px; margin-bottom:4px;
}
.attach-chip{
  display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono', monospace; font-size:0.74rem;
  color:var(--accent2); background:rgba(123,97,255,0.10); border:1px solid rgba(123,97,255,0.3);
  padding:3px 10px; border-radius:999px; margin:2px 6px 2px 0;
}

/* scrollbar */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--border); border-radius:6px; }
</style>
""",
    unsafe_allow_html=True,
)


def render_sentinel(state="standby"):
    """Purely presentational — draws the Sentinel character + status pill. No effect on data/logic."""
    ring_cls = "sentinel-ring fast" if state == "thinking" else "sentinel-ring"
    core_cls = {"standby": "sentinel-core dim", "armed": "sentinel-core", "thinking": "sentinel-core think"}[state]
    status_cls = "sentry-status armed" if state != "standby" else "sentry-status standby"
    status_label = {"standby": "STANDBY", "armed": "ARMED", "thinking": "ANALYZING"}[state]
    return f"""
    <div class="sentry-hero-inner">
      <div class="sentry-brand">
        <div class="sentinel-wrap">
          <div class="{ring_cls}"></div>
          <div class="{core_cls}"></div>
          <div class="sentinel-pupil"></div>
        </div>
        <div>
          <p class="sentry-title">SENTRY</p>
          <p class="sentry-sub">Your AI threat-intel analyst. Type a question, paste a CVE ID, or attach a PDF advisory — right in the chat below.</p>
        </div>
      </div>
      <div class="{status_cls}"><span class="sentry-dot"></span> {status_label}</div>
    </div>
    """


kb_ready = bool(st.session_state.chunks)
st.markdown(f'<div class="sentry-hero">{render_sentinel("armed" if kb_ready else "standby")}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API key setup (sidebar) — access + live mission stats
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
    st.markdown('<p class="console-label">📊 Mission Stats</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mission-stat"><span>Chunks indexed</span><b>{len(st.session_state.chunks)}</b></div>
        <div class="mission-stat"><span>Sources loaded</span><b>{len(set(st.session_state.sources))}</b></div>
        <div class="mission-stat"><span>Messages</span><b>{len(st.session_state.messages)}</b></div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<p class="console-label">📎 How to feed it</p>', unsafe_allow_html=True)
    st.caption("Use the 📎 clip icon in the chat box to attach PDF advisories, or just type a CVE ID (e.g. CVE-2024-3400) and send — Sentry fetches it from NVD automatically.")

    st.divider()
    if st.button("🗑️ Clear knowledge base"):
        st.session_state.pop("index", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("sources", None)
        st.success("Cleared.")

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
# Main chat interface
# ---------------------------------------------------------------------------
st.markdown('<p class="console-label">💬 Interrogate the intel</p>', unsafe_allow_html=True)

if st.session_state.chunks:
    st.markdown(
        f'<span class="kb-pill">🟢 ONLINE — {len(st.session_state.chunks)} chunks '
        f'· {len(set(st.session_state.sources))} source(s)</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="kb-pill empty">⚪ EMPTY — attach a PDF (📎) or type a CVE ID below to arm the Sentinel</span>',
        unsafe_allow_html=True,
    )

AVATARS = {"user": "🧑\u200d💻", "assistant": "🛡️"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        st.markdown(msg["content"])

# Single ChatGPT/Claude/Gemini-style input: type a question, paste a CVE ID,
# and/or attach PDF advisories with the built-in 📎 clip icon — all from one box.
prompt = st.chat_input(
    "Ask a question, paste a CVE ID (e.g. CVE-2024-3400), or attach a PDF...",
    accept_file="multiple",
    file_type=["pdf"],
)

if prompt:
    query_text = (prompt.text or "").strip()
    attached_files = prompt.files or []

    if not client:
        st.error("Please enter a Groq API key in the sidebar first.")
    else:
        ingestion_notes = []

        # 1) Ingest any PDFs attached via the chat's clip icon
        for f in attached_files:
            already_added = f.name in st.session_state.sources
            if not already_added:
                with st.spinner(f"Processing {f.name}..."):
                    text = extract_pdf_text(f.read())
                    n = add_to_index(text, f.name)
                ingestion_notes.append(f'📎 Added **{n}** chunks from `{f.name}`')

        # 2) Auto-detect and fetch any CVE IDs typed directly in the message
        found_cves = sorted(set(m.upper() for m in CVE_PATTERN.findall(query_text)))
        for cve in found_cves:
            if cve in st.session_state.sources:
                continue
            with st.spinner(f"Fetching {cve} from NVD..."):
                try:
                    result = fetch_cve_from_nvd(cve)
                except Exception as e:
                    result = None
                    ingestion_notes.append(f"⚠️ NVD lookup failed for `{cve}`: {e}")
            if result:
                text, cvss = result
                n = add_to_index(text, cve)
                ingestion_notes.append(f'📡 Added **{cve}** — {severity_chip_html(cvss)} · {n} chunk(s)')
            elif result is None:
                ingestion_notes.append(f"⚠️ No NVD data found for `{cve}`")

        # Render the user's turn (attachments shown as chips + their typed text)
        st.session_state.messages.append({"role": "user", "content": query_text or "*(attachment only)*"})
        with st.chat_message("user", avatar=AVATARS["user"]):
            if attached_files:
                chips = "".join(f'<span class="attach-chip">📎 {f.name}</span>' for f in attached_files)
                st.markdown(chips, unsafe_allow_html=True)
            if query_text:
                st.markdown(query_text)

        # Assistant turn: show ingestion results, then answer if a question was asked
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            for note in ingestion_notes:
                st.markdown(note, unsafe_allow_html=True)

            answer = None
            if query_text:
                if not st.session_state.chunks:
                    st.error("Add at least one document or CVE before asking a question.")
                else:
                    st.markdown(
                        '<div class="thinking-strip">'
                        '<div class="sentinel-wrap" style="width:26px;height:26px;">'
                        '<div class="sentinel-ring fast" style="border-width:1px;"></div>'
                        '<div class="sentinel-core think" style="inset:5px;"></div>'
                        '</div> Sentinel is scanning the knowledge base…</div>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Drafting a grounded answer..."):
                        results = retrieve(query_text)
                        answer = answer_question(client, query_text, results)
                        st.markdown(answer)
                        with st.expander("📎 Sources used"):
                            for chunk, src in results:
                                st.markdown(f'<span class="src-chip">{src}</span>', unsafe_allow_html=True)
                                st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))
            elif not ingestion_notes:
                st.markdown("Attach a PDF or type a CVE ID to get started.")

        assistant_content = "\n\n".join(ingestion_notes) if ingestion_notes else ""
        if answer:
            assistant_content = (assistant_content + "\n\n" + answer).strip()
        if assistant_content:
            st.session_state.messages.append({"role": "assistant", "content": assistant_content})
