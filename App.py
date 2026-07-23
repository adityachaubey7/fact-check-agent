"""
Fact-Check Agent — a "Truth Layer" for marketing PDFs.

Pipeline:
  1. Extract text from an uploaded PDF.
  2. Extract verifiable claims (stats, dates, financial/technical figures).
  3. Search the live web for each claim.
  4. Judge each claim as Verified / Inaccurate / False, with the correct
     real-world fact and supporting sources.

Works in two modes:
  - LLM mode (recommended): if an ANTHROPIC_API_KEY is configured, Claude
    reads each claim + live search snippets and returns a structured verdict.
  - Heuristic mode (no API key needed): falls back to numeric/keyword
    matching between the claim and the search snippets, so the app still
    runs and deploys for free with zero paid keys.
"""

import io
import json
import re
import time

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from ddgs import DDGS

st.set_page_config(page_title="Fact-Check Agent", page_icon="🕵️", layout="wide")

# --------------------------------------------------------------------------
# Config / optional LLM client
# --------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        # No secrets.toml at all is fine — app runs in heuristic mode.
        return default


ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY", "")
MODEL_NAME = _get_secret("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

llm_client = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic

        llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        llm_client = None

MAX_CLAIMS = 15  # cap so a single run stays fast and cheap


# --------------------------------------------------------------------------
# Step 1: PDF -> text
# --------------------------------------------------------------------------

def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# --------------------------------------------------------------------------
# Step 2: text -> candidate claims
# --------------------------------------------------------------------------

# A sentence is a "claim" candidate if it contains a number, %, currency,
# a year, or a stat-flavoured keyword. This is a fast, model-free first pass
# that keeps the app deployable even with no API key at all.
NUMERIC_PATTERN = re.compile(
    r"(\$\s?\d[\d,\.]*\s?(?:billion|million|trillion|bn|m|k)?"
    r"|\d[\d,\.]*\s?%"
    r"|\b(19|20)\d{2}\b"
    r"|\d[\d,\.]*\s?(?:billion|million|trillion|users|customers|employees|"
    r"countries|languages|times|x)\b)",
    re.IGNORECASE,
)

STAT_KEYWORDS = re.compile(
    r"\b(grew|growth|increase[d]?|decrease[d]?|revenue|valuation|valued|"
    r"raised|funding|market share|founded|launched|acquired|users|"
    r"customers|employees|percent|according to|study|report|survey)\b",
    re.IGNORECASE,
)


def split_sentences(text: str):
    # Cheap sentence splitter; good enough for marketing/report style PDFs.
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def extract_claims(text: str):
    claims = []
    for sentence in split_sentences(text):
        if NUMERIC_PATTERN.search(sentence) or STAT_KEYWORDS.search(sentence):
            claims.append(sentence)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for c in claims:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:MAX_CLAIMS]


def llm_refine_claims(raw_text: str):
    """Optional: ask Claude to pull the cleanest, most check-worthy claims."""
    if not llm_client:
        return None
    prompt = f"""Read the marketing/report text below and extract up to {MAX_CLAIMS}
distinct, independently verifiable factual claims: specific statistics, dates,
financial figures, or technical numbers. Skip vague or subjective statements.

Return ONLY a JSON array of strings, each one claim, no other text.

TEXT:
\"\"\"{raw_text[:12000]}\"\"\""""
    try:
        resp = llm_client.messages.create(
            model=MODEL_NAME,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        claims = json.loads(raw)
        if isinstance(claims, list) and claims:
            return [str(c) for c in claims[:MAX_CLAIMS]]
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Step 3: live web search per claim
# --------------------------------------------------------------------------

def search_web(query: str, max_results: int = 5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    }
                )
    except Exception as e:
        st.session_state.setdefault("search_errors", []).append(str(e))
    return results


def build_search_query(claim: str) -> str:
    # Trim to the most search-friendly slice of the claim.
    q = re.sub(r"[\"“”]", "", claim)
    return q[:180]


# --------------------------------------------------------------------------
# Step 4a: LLM verdict (preferred)
# --------------------------------------------------------------------------

def llm_verdict(claim: str, evidence: list):
    if not llm_client:
        return None
    evidence_text = "\n".join(
        f"- [{i+1}] {e['title']}: {e['body']} ({e['href']})"
        for i, e in enumerate(evidence)
    ) or "(no search results found)"

    prompt = f"""You are a fact-checking "truth layer" for marketing content.

CLAIM TO CHECK:
"{claim}"

LIVE WEB SEARCH RESULTS FOR THIS CLAIM:
{evidence_text}

Decide the claim's status:
- "Verified" — the search results confirm the figure/date/fact as stated.
- "Inaccurate" — the search results show a different, more current or
  correct figure/date/fact (e.g. an outdated statistic).
- "False" — no evidence supports the claim, or evidence directly
  contradicts it with no reasonable current figure to point to.

Respond ONLY with JSON in exactly this shape:
{{"status": "Verified|Inaccurate|False",
  "real_fact": "the correct fact if Inaccurate/False, else restate the confirmed fact",
  "explanation": "one or two sentence reasoning citing what the evidence showed",
  "source": "best matching URL from the evidence, or empty string"}}"""

    try:
        resp = llm_client.messages.create(
            model=MODEL_NAME,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        data = json.loads(raw)
        return {
            "status": data.get("status", "False"),
            "real_fact": data.get("real_fact", ""),
            "explanation": data.get("explanation", ""),
            "source": data.get("source", ""),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------
# Step 4b: heuristic verdict (zero-API-key fallback)
# --------------------------------------------------------------------------

SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "k": 1e3}


def extract_magnitudes(s: str):
    """Pull (value, unit_category) pairs, scaled to a common magnitude.
    Bare 4-digit numbers that look like years (1900-2099) are treated as
    dates, not statistics, and excluded from magnitude comparison — this is
    what stops "...in 2021" from falsely 'matching' an unrelated "$2 trillion".
    """
    out = []
    for m in re.finditer(
        r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s?(trillion|billion|million|k|percent|%)?",
        s, re.IGNORECASE,
    ):
        num_str = m.group(1)
        unit = (m.group(2) or "").lower()
        if not num_str:
            continue
        try:
            val = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if unit in ("percent", "%"):
            out.append((val, "percent"))
        elif unit in SCALE:
            out.append((val * SCALE[unit], "amount"))
        else:
            if len(num_str) == 4 and 1900 <= int(val) <= 2099:
                continue  # a year, not a statistic
            if val >= 1:  # skip stray tiny fragments
                out.append((val, "amount"))
    return out


def extract_years(s: str):
    return set(re.findall(r"\b(19|20)\d{2}\b", s))


def heuristic_verdict(claim: str, evidence: list):
    if not evidence:
        return {
            "status": "False",
            "real_fact": "No supporting evidence found on the live web for this claim.",
            "explanation": "Search returned no relevant results.",
            "source": "",
        }

    combined_text = " ".join(e["title"] + " " + e["body"] for e in evidence)
    claim_mags = extract_magnitudes(claim)
    evidence_mags = extract_magnitudes(combined_text)

    def close(a, b, tol=0.08):
        return abs(a - b) <= tol * max(abs(a), abs(b), 1)

    matched = any(
        cat_c == cat_e and close(val_c, val_e)
        for val_c, cat_c in claim_mags
        for val_e, cat_e in evidence_mags
    )
    if matched:
        return {
            "status": "Verified",
            "real_fact": claim,
            "explanation": "A matching figure for this claim was found in live search results.",
            "source": evidence[0]["href"],
        }

    if claim_mags and evidence_mags:
        best_val, best_cat = evidence_mags[0]
        return {
            "status": "Inaccurate",
            "real_fact": f"Live sources suggest a different figure — see: {evidence[0]['title']} ({evidence[0]['href']})",
            "explanation": "The claimed figure did not match the figures found in live search results.",
            "source": evidence[0]["href"],
        }

    return {
        "status": "False",
        "real_fact": "No matching figure could be confirmed from live sources.",
        "explanation": "Related results were found but contained no comparable figure to check against.",
        "source": evidence[0]["href"] if evidence else "",
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline(pdf_bytes: bytes, progress_cb=None):
    text = extract_text(pdf_bytes)
    if not text.strip():
        return [], "no_text"

    claims = llm_refine_claims(text) or extract_claims(text)
    if not claims:
        return [], "no_claims"

    rows = []
    for i, claim in enumerate(claims):
        if progress_cb:
            progress_cb(i, len(claims), claim)
        query = build_search_query(claim)
        evidence = search_web(query)
        verdict = llm_verdict(claim, evidence) or heuristic_verdict(claim, evidence)
        rows.append(
            {
                "Claim": claim,
                "Status": verdict["status"],
                "Real Fact / Correction": verdict["real_fact"],
                "Explanation": verdict["explanation"],
                "Source": verdict["source"],
            }
        )
        time.sleep(0.3)  # be polite to the search backend
    return rows, "ok"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🕵️ Fact-Check Agent")
st.caption(
    "Upload a marketing PDF. The agent extracts claims, checks them against "
    "live web data, and flags stats that are outdated or made up."
)

with st.sidebar:
    st.subheader("Mode")
    if llm_client:
        st.success(f"LLM mode active — using {MODEL_NAME}")
    else:
        st.info(
            "Heuristic mode (no ANTHROPIC_API_KEY set).\n\n"
            "Add one in Streamlit secrets for smarter, more nuanced verdicts. "
            "The app fully works without it."
        )
    st.subheader("Legend")
    st.markdown(
        "- 🟢 **Verified** — matches live data\n"
        "- 🟡 **Inaccurate** — outdated / wrong figure\n"
        "- 🔴 **False** — no evidence found"
    )

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    st.write(f"**File:** {uploaded.name}")
    if st.button("Run Fact-Check", type="primary"):
        progress_bar = st.progress(0.0, text="Starting…")
        status_area = st.empty()

        def on_progress(i, n, claim):
            progress_bar.progress((i) / max(n, 1), text=f"Checking claim {i+1}/{n}…")
            status_area.caption(f"🔎 {claim[:120]}")

        rows, state = run_pipeline(uploaded.read(), progress_cb=on_progress)
        progress_bar.progress(1.0, text="Done")

        if state == "no_text":
            st.error("Couldn't extract any text from this PDF (it may be scanned/image-only).")
        elif state == "no_claims":
            st.warning("No verifiable numeric/statistical claims were found in this document.")
        else:
            df = pd.DataFrame(rows)

            c1, c2, c3 = st.columns(3)
            c1.metric("Verified", (df["Status"] == "Verified").sum())
            c2.metric("Inaccurate", (df["Status"] == "Inaccurate").sum())
            c3.metric("False", (df["Status"] == "False").sum())

            def badge(status):
                return {"Verified": "🟢", "Inaccurate": "🟡", "False": "🔴"}.get(status, "⚪")

            st.divider()
            for _, row in df.iterrows():
                with st.expander(f"{badge(row['Status'])} **{row['Status']}** — {row['Claim'][:100]}"):
                    st.write(f"**Claim:** {row['Claim']}")
                    st.write(f"**Real fact / correction:** {row['Real Fact / Correction']}")
                    st.write(f"**Why:** {row['Explanation']}")
                    if row["Source"]:
                        st.write(f"**Source:** {row['Source']}")

            st.divider()
            st.download_button(
                "Download report as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="fact_check_report.csv",
                mime="text/csv",
            )
else:
    st.info("👆 Upload a PDF to begin. Try it with a marketing one-pager or a report containing stats.")
