"""Pure numerical core of the CS-model: no I/O, no GUI, deterministic.

Every public function names the article formula or algorithm step it implements: normalizers,
metrics, operators, neighbours, k strategies, encoders, formulas (1)–(6) and the trace (Steps 1–8,
:mod:`.context`), majorizers and the HAG (Step 9, :mod:`.hag`), the meta-algorithm (Step 12,
:mod:`.meta`) and the model end to end (:mod:`.model`). Property checks follow in M4;
:mod:`.registry` provides the plug-in mechanism.
"""
