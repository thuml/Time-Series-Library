# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | sanity | `DLinear`, `Δ=0`, same-time baseline | Hoister 5-class | macro-F1, balanced_acc | MUST | TODO | base task sanity |
| R002 | M0 | sanity | `DLinear`, `Δ=1`, shifted hard label | Hoister 5-class | macro-F1, balanced_acc | MUST | TODO | verify shifted target path |
| R003 | M1 | baseline | `TimesNet`, `Δ=1`, shifted hard CE | Hoister 5-class | macro-F1, fault_macro_f1 | MUST | TODO | anchor baseline |
| R004 | M2 | main method | `TimesNet`, `Δ=1`, proposed boundary supervision | Hoister 5-class | macro-F1, class9_recall | MUST | TODO | first proof run |
| R005 | M3 | novelty isolation | `TimesNet`, `Δ=1`, shifted focal | Hoister 5-class | macro-F1, class9_f1 | MUST | TODO | compare against focal |
| R006 | M3 | novelty isolation | `TimesNet`, `Δ=1`, shifted sampler+focal | Hoister 5-class | macro-F1, class9_f1 | MUST | TODO | compare against sampler |
| R007 | M4 | transfer | `iTransformer`, `Δ=1`, shifted hard vs proposed | Hoister 5-class | macro-F1, balanced_acc | MUST | TODO | second backbone |
| R008 | M4 | transfer | `DLinear`, `Δ=1`, shifted hard vs proposed | Hoister 5-class | macro-F1, balanced_acc | MUST | TODO | simple backbone |
| R009 | M4 | leakage check | best shifted baseline with `JianSuDuan_ChaoSu` kept | Hoister 5-class | macro-F1 | MUST | TODO | leakage probe |
| R010 | M4 | leakage check | best shifted baseline with `JianSuDuan_ChaoSu` dropped | Hoister 5-class | macro-F1 | MUST | TODO | leakage probe |
| R011 | M5 | ablation | proposed without boundary soft target | Hoister 5-class, `Δ=1` | macro-F1, class9_recall | MUST | TODO | deletion study |
| R012 | M5 | ablation | proposed without transition upweighting | Hoister 5-class, `Δ=1` | macro-F1, class9_recall | MUST | TODO | deletion study |
| R013 | M5 | horizon | shifted hard baseline, `Δ=5` | Hoister 5-class | macro-F1, balanced_acc | MUST | TODO | horizon probe |
| R014 | M5 | horizon | proposed method, `Δ=5` | Hoister 5-class | macro-F1, balanced_acc | MUST | TODO | horizon probe |
| R015 | M5 | robustness | strongest shifted hard baseline, 3 seeds | Hoister 5-class, `Δ=1` | mean/std macro-F1 | NICE | TODO | seed stability |
| R016 | M5 | robustness | proposed method, 3 seeds | Hoister 5-class, `Δ=1` | mean/std macro-F1 | NICE | TODO | seed stability |
