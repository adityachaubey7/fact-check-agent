# Fact-Check Agent 🕵️

A "Truth Layer" for marketing PDFs. Upload a PDF, and the agent:

1. **Extracts** verifiable claims — stats, dates, financial and technical figures.
2. **Verifies** each one against live web search results.
3. **Reports** each claim as ✅ **Verified**, 🟡 **Inaccurate** (outdated/wrong figure), or 🔴 **False** (no supporting evidence).

Built for the CogCulture Product Management Trainee assessment (Part 2).



## How it works

```
PDF upload → text extraction (pypdf)
           → claim extraction (regex heuristics, optionally refined by Claude)
           → live web search per claim (DuckDuckGo via `ddgs`, no API key needed)
           → verdict per claim
                ├─ LLM mode: Claude reads the claim + search snippets → JSON verdict
                └─ Heuristic mode: magnitude-aware number matching between
                   the claim and the search snippets (used automatically if
                   no ANTHROPIC_API_KEY is configured)
           → Streamlit report (status badges, explanation, source, CSV export)
```

The app **runs with zero paid API keys** out of the box (heuristic mode), and automatically
upgrades to LLM-graded verdicts the moment an `ANTHROPIC_API_KEY` is added to Streamlit
secrets — no code changes needed.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the printed local URL, upload a PDF, and click **Run Fact-Check**.

## Deploying (Streamlit Community Cloud — free)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick this repo, branch `main`, and set the main file path to `app.py`.
4. (Optional, recommended) Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
   ```
   This switches the app from heuristic mode to LLM-graded verdicts.
5. Click **Deploy**. You'll get a URL like `https://your-app.streamlit.app`.

The app also deploys the same way on Render or a Vercel Python function — just point
the platform at `app.py` and `requirements.txt`; no other config is required.

## Design notes / trade-offs

- **Claim extraction** starts with a fast regex pass (numbers, %, currency, years,
  stat keywords) so the app never depends on an API key to find claims. If an
  `ANTHROPIC_API_KEY` is set, Claude re-extracts cleaner, deduplicated claims instead.
- **Web search** uses `ddgs` (DuckDuckGo search), which needs no API key or billing —
  important for a trainee-assessment app that has to "just work" when graded.
- **Verdict** logic is deliberately layered: Claude gives nuanced, explained verdicts
  when available; the heuristic fallback does magnitude-aware number comparison
  (normalizing "billion/trillion/percent", and explicitly excluding bare years like
  "2021" from being mistaken for a statistic) so it doesn't false-positive on
  coincidental digit overlap.
- **Claims are capped at 15 per document** (`MAX_CLAIMS` in `app.py`) to keep a single
  run fast and, in LLM mode, inexpensive. Raise it for larger documents.
- **CSV export** is included so a reviewer can archive/share the full report.

## Known limitations

- Scanned/image-only PDFs with no extractable text will report "no text found" —
  OCR is not included (out of scope for this assessment).
- Heuristic mode is best-effort without an LLM; for nuanced or non-numeric claims
  (e.g. qualitative superiority claims), LLM mode gives materially better verdicts.
- DuckDuckGo search can rate-limit high-volume/rapid use; the app adds a small
  delay between claims and gracefully returns "no evidence found" if a search fails.

## Repo structure

```
app.py             # the whole app — Streamlit UI + pipeline
requirements.txt   # pinned dependencies
README.md          # this file
```
