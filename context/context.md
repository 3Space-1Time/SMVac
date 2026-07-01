# Project Context

## Current Objective
Provide a permanent cross-agent context system for the Threshold project to prevent token limit amnesia and ensure continuity across different AI coding agents.

## Project Overview
Threshold is a C++ and Python project analyzing the stability of the Standard Model vacuum using exact ODE solvers to compute the bounce action. The project aims to rigorously eliminate numerical artifacts, specifically proving that the "Fubini-Lipatov sandwich" (a previously predicted instability band) is an artifact of the conformal approximation and does not exist when evaluated numerically with exact RG evolution and derivatives.

## Important Decisions
- Adopted exact RK4 ODE integration with analytical derivatives to eliminate finite-difference errors.
- Added dynamic step-scaling and a hard cutoff (100,000 max GSS steps) to manage slow convergence in flat potential regions.
- Successfully completed a dense parameter grid evaluation (14,641 points) over $M_h \in [110, 140]$ GeV and $M_t \in [160, 185]$ GeV.

## Architecture
- **C++ Solvers:** `solver_ode_grid.cpp` handles the core integration using exact potentials and analytic derivatives.
- **Python Drivers:** Scripts like `run_ode_grid.py` distribute execution across available cores in chunks.
- **Plotting:** Scripts like `plot_ode_grid.py` aggregate CSV outputs to generate standard stability phase diagrams.
- **Data:** Results stored in `results/ode_grid_data.csv`.

## Dataset Information
- 121x121 points grid ($M_h \in [110, 140]$, $M_t \in [160, 185]$) yielding 14,641 physics data points evaluating bounce actions and virial ratios.

## Files Modified
- `context/context.md` (new)
- `context/context.yaml` (new)

## Commands Executed
- (Various C++ and Python commands during previous sessions to compile, execute, and plot the ODE grid).

## Results
- The Fubini-Lipatov sandwich has been definitively proven to be a numerical artifact. The exact ODE solver resolves the entire low $M_h$ metastability region smoothly. The final phase diagram (`ode_stability_plot.png`) accurately reflects standard theoretical expectations.

## Errors Encountered
- Finite-difference derivative estimates were inaccurate at large radii.
- The solver previously stalled/hung on very flat potential regions.
- Python chunk combiners faced file-locking `PermissionError` when deleting overlapping files while scripts were still running.

## Fixes Applied
- Implemented analytical $dV/d\phi$.
- Added iteration limits and dynamic target step size (`std::min(dr_target, 0.1 * r + 1.0)`) to the ODE solver.
- Allowed scripts to overwrite old chunks and cleanly finish aggregation.

## Next Steps
- Integrate the findings and the generated phase diagram into the LaTeX manuscript.
- Finalize the write-up and verify formatting.

## Pending Tasks
- Integrate into LaTeX (`paper/main.tex`).
- Complete the 1M-point exact ODE grid execution.

## Agent Handoff Notes

Agent: Antigravity
Timestamp: 2026-06-19T10:28:00+05:30
Task Completed: Re-configured C++ solver and python script to calculate the full 1M points grid. Launched execution in the background.
Current State: 1M point grid evaluating. `results/ode_grid_data_chunk_*.csv` being written.
Known Issues: None at present.
Next Recommended Action: Await task completion, combine dataset, plot the 1M point diagram, and proceed with writing/updating the LaTeX manuscript.

## Conversation Summary
The previous sessions successfully designed, tested, and implemented a robust numerical ODE solver for the bounce action. The solver was deployed across the target parameter space in parallel chunks, resolving known hangs. The final phase diagram confirmed the elimination of the Fubini-Lipatov artifact. The user then requested the initialization of this permanent context tracking system. We then began a 1,000,000-point full exact ODE grid scan.

## Latest Update
2026-06-19: Launched the full 1M-point exact ODE grid scan. Wait for the background task to complete.
