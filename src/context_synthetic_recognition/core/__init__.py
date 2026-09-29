"""Pure numerical core of the CS-model: no I/O, no GUI, deterministic.

Every public function names the article formula or algorithm step it implements. The modules
(normalizers, metrics, operators, neighbours, k strategies, encoders, formulas (1)–(6), majorizers,
HAG, meta-algorithm, model, property checks, trace) arrive with milestones M2–M4; :mod:`.registry`
provides the plug-in mechanism they register with.
"""
