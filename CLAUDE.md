# context-synthetic-recognition — project memory for Claude Code

Python implementation (library + CLI + desktop GUI) of the context-synthetic model of recognition algorithms (CS-model)
from the author's draft article, built on the full Excel experiment.

Start here: `docs/handoff/PROMPT_FOR_CLAUDE_CODE.md` (task), `docs/handoff/CONTEXT_HANDOFF.md` (authoritative decisions and
golden values), `docs/handoff/CHAT_TRANSCRIPT.md` (design conversation).

Non-negotiables (details in CONTEXT_HANDOFF.md):
- Specification = `resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [...] - Opus 5.5.xlsx`; its Validation sheet values are acceptance tests (tolerance 1e-9).
- Permitted k: odd, k_min = 3 fixed, k_max = 2·min_i|K_i| − 3 from the training classes — never hard-coded.
- Distances rounded to 10 decimals; ties → smaller original index; normalization fitted on training data only.
- A new object is represented and classified without its class (Theorem); LOO re-fits the whole pipeline per fold.
- Two template deviations are switches (default = template): running vs final class centres in θ/γ; STEP 4 majorizer applied twice vs once. Highlight them in docs and GUI.
- Never modify `../hag-regularized-stacking-boosting-meta` (template project, read-only), `resources/experiments/` or `docs/handoff/`.
- Record decisions in `docs/DECISIONS.md`; ask the author before changing anything that alters numerical results.
