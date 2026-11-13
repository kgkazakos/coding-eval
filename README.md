# coding-eval

**Measures LLM thematic coding accuracy and agreement rates against human coders.**

`coding-eval` is a lightweight CLI tool that quantifies how well a language model reproduces human thematic coding decisions. It computes per-code Cohen's Kappa, precision, recall, and F1 — giving qualitative researchers a principled, reproducible way to validate LLM-assisted analysis before using it in production.

---

## The problem it solves

You've started using an LLM to assist with thematic coding. It seems to match your codes most of the time — but how close is *most of the time*? Without a rigorous agreement metric, you're guessing. `coding-eval` turns that gut feel into a defensible number.

---

## Features

- **Per-code Cohen's Kappa** — binary agreement computed for each code in your scheme, aggregated by support (weighted) and macro average
- **Exact set agreement** — proportion of segments where LLM codes match the human set exactly
- **Precision, Recall, F1** — per code, plus macro and weighted aggregates
- **A–F grade** — based on Landis & Koch (1977) Kappa interpretation thresholds
- **Three workflows** — `eval` (LLM re-codes live), `compare` (two pre-coded files), `batch` (directory sweep)
- **LLM on by default** — `--llm-backend openai` is the default; pass `--llm-backend none` if you have pre-coded files
- **Model-agnostic** — rule-based path runs fully without API keys; LLM enrichment via OpenAI, Anthropic, or Gemini
- **JSON output** — `--json-output` for CI integration and downstream analysis
- **DSPy `ChainOfThought`** — structured reasoning for both coding and narrative generation

---

## Install

```bash
pip install coding-eval
```

With LLM support:

```bash
pip install "coding-eval[llm,openai]"      # OpenAI
pip install "coding-eval[llm,anthropic]"   # Anthropic
pip install "coding-eval[llm,gemini]"      # Gemini
```

---

## Usage

### `eval` — LLM re-codes your segments, then computes agreement

```bash
# Uses OpenAI by default
coding-eval eval my_human_codes.json

# Choose a different backend
coding-eval eval my_human_codes.json --llm-backend anthropic

# Already have llm_codes in your file? Skip the API call
coding-eval eval my_coded_pairs.json --llm-backend none

# Output JSON for CI or downstream scripts
coding-eval eval my_coded_pairs.json --llm-backend none --json-output
```

### `compare` — compare two separate coding files

```bash
coding-eval compare human_codes.json llm_codes.json --no-narrative
```

### `batch` — sweep a directory of pre-coded files

```bash
coding-eval batch ./coded_sessions/ --pattern "*.json"
```

---

## Input format

**JSON:**
```json
[
  {
    "segment_id": "s01",
    "text": "The checkout is really fast",
    "codes": ["satisfaction", "usability"],
    "llm_codes": ["satisfaction", "usability"]
  }
]
```
The `llm_codes` field is optional — required only when using `--llm-backend none`.

**CSV:**
```
segment_id,text,codes,llm_codes
s01,The checkout is really fast,satisfaction;usability,satisfaction;usability
```
Codes within a cell are semicolon-separated.

---

## Grade scale

| Grade | κ range     | Interpretation                  |
|-------|-------------|----------------------------------|
| A     | ≥ 0.80      | Almost perfect agreement         |
| B     | 0.60 – 0.79 | Substantial agreement            |
| C     | 0.40 – 0.59 | Moderate agreement               |
| D     | 0.20 – 0.39 | Fair agreement                   |
| F     | < 0.20      | Slight or poor agreement         |

Based on Landis & Koch (1977). Scores below 0.60 typically warrant coder calibration before proceeding.

---

## LLM backends

| Flag | Default model | Key required |
|------|--------------|-------------|
| `openai` (default) | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `anthropic/claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini/gemini-3.6-flash` | `GEMINI_API_KEY` |
| `none` | — | — |

All backends route through [LiteLLM](https://github.com/BerriAI/litellm) via DSPy, so any LiteLLM-compatible model string works. To use a stronger model, edit `BACKENDS` in `coding_eval/coder.py` — swap `openai/gpt-4o-mini` for `openai/gpt-4o` or `anthropic/claude-opus-4-6` for higher-stakes calibration runs.

---

## Part of the CPR Ecosystem

`coding-eval` is Tool 18 of 40 in the [CPR Orbital series](https://github.com/kgkazakos) — a portfolio of open-source Python tools for AI-native product research infrastructure, shipping continuously through 2026.

**Related tools:**
- [`qual-sniffer`](https://github.com/kgkazakos/qual-sniffer) — detects methodological weaknesses in discussion guides
- [`research-rubric`](https://github.com/kgkazakos/research-rubric) — evaluates qualitative research quality across 8 dimensions
- [`bias-audit`](https://github.com/kgkazakos/bias-audit) — detects interviewer bias patterns across transcript corpora
- [`uxr-bench`](https://github.com/kgkazakos/uxr-bench) — benchmark dataset for LLM performance on qualitative research tasks
- [`saturation-meter`](https://github.com/kgkazakos/saturation-meter) — estimates thematic saturation from coded transcripts

Part of [Computational Product Research (CPR)](https://github.com/kgkazakos) — an independent discipline for AI-native product research.

---

## License

MIT © Kostas Kazakos
