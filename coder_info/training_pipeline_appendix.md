## Next actions before changes 20–33

1. **Finish change_19** (rewrite consistency) — implement batch pairing and symmetric answer-masked KL (see `fix_01_*`).
2. **Guardrails** — enforce config/tag parity & atomicity; wire a run manifest for reproducibility (`fix_02`, `fix_03`, `fix_13`).
3. **On-policy everywhere** — elevate on-policy decode from smoke into the main loop; log RL stats and gate coverage (`fix_05`, `fix_06`).
4. **Budget KPI** — automate quality-vs-budget sweeps; ship CSV per run (`fix_07`).
5. **Calibration parity** — persist plan thresholds/budget clip; apply at serve (`fix_08`).
6. **Reward correctness** — token-basis budget penalty; dataset strictness (`fix_09`, `fix_10`).
7. **Answer invariance** — verify that strategy hints never alter the final answer (`fix_11`).

When these land, proceed with strategy-oriented heads (changes 20–33): cognitive ops, strategy tags, subgoal planner, invariants/NLI checks, analogical structure, elimination grids, reflection/program-of-thought.  
All subsequent features must preserve **serve/eval parity**, **hidden CoT**, **budget discipline**, and **leakage≈0**.


