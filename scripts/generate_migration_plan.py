import os

files = [
    "ABSTRACT_PLAN.md", "AGENTS.md", "AUDIT_REBUTTAL.md", "CLAIM_AUDIT.md", "CLAIM_IMPACT.md", "FIGURE_PLAN.md", "FIGURE_REVIEW.md", "MISSING_ANALYSIS.md", "NEW_OUTLINE.md", "ODE_VERIFICATION_PLAN.md", "PAPER_REVIEW.md", "POTENTIAL_DISCREPANCY_REPORT.md", "POTENTIAL_STRUCTURE_REPORT.md", "PROJECT_AUDIT.md", "SANDWICH_ANALYSIS.md", "SANDWICH_REFEREE_TEST.md", "SANDWICH_STORY.md", "SOLVER_VERIFICATION.md", "STORY_ANALYSIS.md", "_audit_exp_region.py", "_audit_final.py", "_audit_refined.py", "_audit_sandwich.py", "_check_chunk35.py", "_check_chunk43846.py", "_check_chunk50752.py", "_check_chunks.py", "_error_check.py", "_inspect_diagnostic.py", "_sandwich_analysis.py", "_sandwich_check.py", "_sandwich_detail.py", "_sandwich_extent.py", "_sandwich_map.py", "_sandwich_plot.py", "analyze_maxsteps.py", "analyze_potential.py", "context/context.md", "critique_prompt_for_claude.md", "debug_shoot.cpp", "deriv_solver.cpp", "docs/error_analysis_report.md", "docs/research_paper_draft.md", "docs/stability_analysis_report.md", "docs/stability_code_report.md", "dummy.cpp", "find_interval.cpp", "generate_audit.py", "metastable_points_mh5.md", "paper/figures/analytical_stability_closeup.png", "paper/figures/analytical_stability_plot.png", "paper/figures/buttazzo_plot.png", "paper/figures/c6_phase_diagram.png", "paper/figures/c6_sweep_plot.png", "paper/figures/error_S_scatter.png", "paper/figures/error_boundary_reclassification.png", "paper/figures/error_discrepancy_heatmap.png", "paper/figures/error_golden_section.png", "paper/figures/error_quadrature_convergence.png", "paper/figures/error_residual_histogram.png", "paper/figures/error_rk4_convergence.png", "paper/figures/error_vs_lambda.png", "paper/figures/numerical_stability_closeup.png", "paper/figures/numerical_stability_contours.png", "paper/figures/numerical_stability_plot.png", "paper/figures/overlay_closeup_plot.png", "paper/figures/overlay_plot.png", "paper/main.tex", "patch2.py", "patch_deriv.py", "plot_analytical.py", "plot_bisection.py", "plot_numerical.py", "plot_numerical_contours.py", "plot_ode_grid.py", "plot_overlay.py", "plot_traces.py", "potential_analysis_rge.cpp", "potential_scan.cpp", "prompt_for_claude.md", "reformulate_prompt_for_claude.md", "data/analytical_data.csv", "data/analytical_dense_closeup.csv", "figures/analytical_stability_closeup.png", "figures/analytical_stability_plot.png", "data/bisection_converged.csv", "data/bisection_overshoot.csv", "figures/bisection_plot.png", "data/bisection_undershoot.csv", "data/deriv_comparison.csv", "data/derivative_comparison.csv", "data/derivative_ode_results.csv", "data/diagnostic_output.csv", "figures/diagnostic_overlay.png", "data/epsilon_test_results.csv", "data/instrumented_trajectories.csv", "data/max_steps_diag.csv", "data/numerical_data.csv", "data/numerical_dense_closeup.csv", "figures/numerical_stability_closeup.png", "figures/numerical_stability_contours.png", "figures/numerical_stability_plot.png", "data/ode_convergence_raw.csv", "figures/ode_dphi_dr.png", "data/ode_grid_data.csv", "data/ode_grid_data_chunk_0.csv", "data/ode_grid_data_chunk_10502.csv", "data/ode_grid_data_chunk_15753.csv", "data/ode_grid_data_chunk_21004.csv", "data/ode_grid_data_chunk_26255.csv", "data/ode_grid_data_chunk_31506.csv", "data/ode_grid_data_chunk_36757.csv", "data/ode_grid_data_chunk_42008.csv", "data/ode_grid_data_chunk_47259.csv", "data/ode_grid_data_chunk_5251.csv", "data/ode_grid_data_chunk_52510.csv", "data/ode_grid_data_chunk_57761.csv", "figures/ode_phi_r.png", "data/ode_profile.csv", "data/ode_profile_105.csv", "data/ode_profile_106.csv", "data/ode_profile_110.csv", "figures/ode_stability_plot.png", "figures/ode_stability_plot_partial.png", "data/ode_validation_raw.csv", "figures/overlay_closeup_plot.png", "figures/overlay_plot.png", "data/partial_ode_grid.csv", "figures/partial_ode_plot.png", "figures/potential_discrepancy.png", "data/potential_scan.csv", "figures/potential_structure_plot.png", "data/potential_verification_grid.csv", "data/profile_ana.csv", "data/profile_fd.csv", "figures/sandwich_neg_fraction.png", "figures/sandwich_panel.png", "figures/sandwich_zoom_Mh100.png", "figures/stability_closeup_plot.png", "figures/stability_plot.png", "data/stress_test_results.csv", "data/sweep_summary.csv", "data/trace_0.csv", "data/trace_1.csv", "data/trace_2.csv", "figures/trace_plot.png", "data/verify_stationary_points.csv", "run_analytical.py", "run_bisection.cpp", "run_dense.py", "run_numerical.py", "run_ode_grid.py", "solver_analytical.cpp", "solver_analytical_dense.cpp", "solver_canonical.cpp", "solver_canonical_A.cpp", "solver_canonical_B.cpp", "solver_diagnostic.cpp", "solver_instrumented.cpp", "solver_maxsteps_diag.cpp", "solver_numerical.cpp", "solver_numerical_dense.cpp", "solver_ode_diagnostic.cpp", "solver_ode_grid.cpp", "src/cosmology/higgs_inflation.py", "src/cosmology/landscape_bertini_solver.py", "src/cosmology/landscape_gradient_flow.py", "src/plotting/plot_error_scaling.py", "src/plotting/plot_interim.py", "src/plotting/plot_phase_diagram.py", "src/plotting/plot_precision_comparison.py", "src/plotting/plot_scatter.py", "src/sm_stability/bounce_action_solver_unopt.cpp", "src/sm_stability/error_scaling_test.cpp", "src/sm_stability\error_scaling_test2.cpp", "src/sm_stability/fix.py", "src/sm_stability/fix_grid.py", "src/sm_stability/patch_fix.py", "src/sm_stability/patch_main.py", "src/sm_stability/patch_main2.py", "src/sm_stability/patch_opt.py", "src/sm_stability/patch_physics.py", "src/sm_stability/sm_rge_solver.cpp", "src/sm_stability/test.py", "src/sm_stability/test_exact.cpp", "src/sm_stability/test_interp.cpp", "src/sm_stability/test_opt.py", "src/sm_stability/test_rge.py", "src/sm_stability/test_solver.cpp", "src/sm_stability/test_solver2.cpp", "src/sm_stability/vacuum_precision_test.cpp", "src/sm_stability/vacuum_precision_test_quad.cpp", "stationary_finder.cpp", "stress_solver.cpp", "sweep_interval.cpp", "test_dv.cpp", "test_nan.cpp", "test_v.cpp", "validation_solver.cpp", "verify_continuity.py", "verify_potential.cpp", "write_test_nan.py"
]

def classify_group(f):
    if f.endswith('tex') or f.endswith('bib') or 'paper/figures' in f: return 'Paper'
    if 'analytical' in f and f.endswith('.cpp'): return 'Fubini-Lipatov approximation'
    if 'numerical' in f and f.endswith('.cpp'): return 'Three-loop RGE evolution / Exact Integration'
    if f in ['solver_canonical.cpp', 'run_bisection.cpp', 'sweep_interval.cpp']: return 'Canonical ODE solver'
    if f in ['stationary_finder.cpp']: return 'Stationary-point analysis'
    if f in ['verify_potential.cpp', 'potential_analysis_rge.cpp', 'analyze_potential.py', 'test_v.cpp', 'test_dv.cpp']: return 'Potential verification'
    if f in ['find_interval.cpp', '_sandwich_analysis.py', '_sandwich_check.py', '_sandwich_detail.py', '_sandwich_extent.py', '_sandwich_map.py', '_sandwich_plot.py']: return 'Sandwich region diagnostics'
    if 'diagnostic' in f or 'instrumented' in f or 'maxsteps' in f or 'stress' in f or 'debug' in f or 'validation_solver' in f or 'deriv_solver' in f: return 'Bounce diagnostics / Numerical stress tests'
    if 'plot' in f: return 'Plotting'
    if f.endswith('.md'): return 'Markdown Reports'
    if f.startswith('results/'): return 'Data & Figures (Results)'
    if 'sm_stability' in f or 'cosmology' in f: return 'Obsolete prototyping code'
    if f.startswith('run_'): return 'Phase-boundary scan wrappers'
    return 'Other scripts'

def classification_status(f):
    # Determine KEEP, ARCHIVE, IGNORE
    if f.startswith('src/sm_stability/') or f.startswith('src/cosmology/'): return 'ARCHIVE'
    if f.startswith('results/'):
        if f.endswith('.csv'): return 'KEEP'
        return 'IGNORE' # PNGs can be regenerated
    if f.endswith('.md') and '_' in f and f.isupper(): return 'KEEP'
    if f.startswith('paper/'): return 'KEEP'
    if f in ['solver_numerical.cpp', 'solver_analytical.cpp', 'run_numerical.py', 'run_analytical.py', 'plot_numerical.py', 'plot_analytical.py', 'plot_overlay.py', 'plot_numerical_contours.py']: return 'KEEP'
    if f in ['run_bisection.cpp', 'sweep_interval.cpp', 'stationary_finder.cpp', 'verify_potential.cpp', 'solver_canonical.cpp']: return 'KEEP'
    if f.endswith('.py') and f.startswith('_'): return 'ARCHIVE' # temporary analysis scripts
    if 'debug' in f or 'test_' in f or 'dummy' in f or 'patch' in f: return 'IGNORE'
    if 'diagnostic' in f or 'instrumented' in f or 'deriv_solver' in f or 'validation_solver' in f or 'stress_solver' in f: return 'ARCHIVE'
    return 'IGNORE'

def version_status(f):
    if f == 'solver_numerical.cpp': return 'FINAL'
    if f == 'solver_analytical.cpp': return 'FINAL'
    if f == 'solver_canonical.cpp': return 'EXPERIMENTAL'
    if f == 'run_bisection.cpp': return 'FINAL'
    if f == 'stationary_finder.cpp': return 'FINAL'
    if f == 'verify_potential.cpp': return 'VALIDATION'
    if f == 'sweep_interval.cpp': return 'VALIDATION'
    if 'diagnostic' in f or 'instrumented' in f or 'stress' in f or 'deriv' in f: return 'DEBUGGING'
    if f.startswith('src/sm_stability/'): return 'OBSOLETE'
    if f.endswith('.md'): return 'DOCUMENTATION'
    if 'run_' in f and f.endswith('.py'): return 'FINAL'
    if 'plot_' in f and f.endswith('.py'): return 'FINAL'
    return 'INTERMEDIATE'

with open('SMVAC_MIGRATION_PLAN.md', 'w', encoding='utf-8') as out:
    out.write('# SMVac Migration Plan\n\n')
    
    out.write('## Task 1 — Original Research Files\n\n')
    out.write('The following is the exhaustive list of source files, headers, scripts, reports, and data outputs authored or generated natively during the research process (excluding all third-party libraries, toolchains, `.git`, `venv`, and `external/` files):\n\n')
    for f in sorted(files):
        out.write(f'* `{f}`\n')
    
    out.write('\n## Task 2 — Grouped by Scientific Purpose\n\n')
    groups = {}
    for f in sorted(files):
        g = classify_group(f)
        if g not in groups: groups[g] = []
        groups[g].append(f)
        
    for g, flist in groups.items():
        out.write(f'### {g}\n')
        for f in flist:
            out.write(f'* `{f}`\n')
        out.write('\n')

    out.write('## Task 3 — Final vs. Intermediate Versions\n\n')
    out.write('| File | Status | Reason |\n')
    out.write('|------|--------|--------|\n')
    for f in sorted(files):
        if not f.endswith('.cpp') and not f.endswith('.py'): continue
        if f.startswith('results/'): continue
        st = version_status(f)
        if st == 'FINAL': reason = "Production pipeline component used for final phase diagram or physics result."
        elif st == 'VALIDATION': reason = "Used to explicitly verify theoretical formulations or equations of motion."
        elif st == 'DEBUGGING': reason = "Instrumented copy used strictly to extract internal state during explosions."
        elif st == 'OBSOLETE': reason = "Legacy prototyping code superseded by root solvers."
        elif st == 'EXPERIMENTAL': reason = "Proofs of concept like canonical solver without integration scaling."
        else: reason = "One-off script or intermediate patch."
        out.write(f'| `{f}` | **{st}** | {reason} |\n')
        
    out.write('\n## Task 4 — Scientific Results Inventory\n\n')
    out.write('''| Title | Code Used | Figures | Numerical Outputs | Pub Quality | GitHub Quality |
|-------|-----------|---------|-------------------|-------------|----------------|
| **Three-loop RGE evolution** | `solver_numerical.cpp` | `error_vs_lambda.png` | `numerical_data.csv` | Yes | Yes |
| **Vacuum stability phase diagram** | `run_numerical.py` | `numerical_stability_plot.png`, `numerical_stability_contours.png` | `numerical_data.csv` | Yes | Yes |
| **FL approximation** | `run_analytical.py` | `analytical_stability_plot.png` | `analytical_data.csv` | Yes | Yes |
| **High-field approximation validation** | `plot_overlay.py` | `overlay_plot.png`, `overlay_closeup_plot.png` | - | Yes | Yes |
| **Sandwich analysis** | `find_interval.cpp` | `sandwich_panel.png` | - | Yes | Yes |
| **Stationary points** | `stationary_finder.cpp` | `potential_structure_plot.png` | `verify_stationary_points.csv` | Yes | Yes |
| **Potential reconstruction** | `verify_potential.cpp` | `potential_discrepancy.png` | `potential_verification_grid.csv` | Yes | Yes |
| **ODE convergence investigation** | `sweep_interval.cpp` | `trace_plot.png` | `sweep_summary.csv`, `trace_*.csv` | Yes (Appendix) | Yes |
| **Canonical bounce** | `run_bisection.cpp` | `bisection_plot.png` | `bisection_converged.csv` | Yes | Yes |
| **Benchmark Mh=5 Mt=105** | `run_bisection.cpp` | `bisection_plot.png` | `bisection_converged.csv` | Yes | Yes |
| **Stress tests** | `stress_solver.cpp` | `error_residual_histogram.png` | `stress_test_results.csv` | No | No (Internal) |
''')

    out.write('\n## Task 5 — SMVac Repository Contents & Migration Map\n\n')
    out.write('The `SMVac` repository will be structured cleanly to accompany the publication. No compiled binaries, external libraries, or temporary files are included.\n\n')
    
    out.write('### Proposed `SMVac/` Structure\n')
    out.write('''```
SMVac/
├── README.md
├── LICENSE
├── docs/                   # Finalized markdown reports
├── src/
│   ├── numerical/          # Exact numerical ODE solver
│   ├── analytical/         # Conformal approximation solver
│   └── diagnostics/        # Stationary finder, bisection, interval sweeps
├── python/
│   ├── runners/            # Parallel grid execution
│   └── plotting/           # Figure generation
├── data/                   # Output CSVs (LFS)
├── figures/                # Output PNGs
└── paper/                  # LaTeX manuscript
```\n\n''')

    out.write('### Action Plan\n\n')
    out.write('| File | Action | New Location (if kept) |\n')
    out.write('|------|--------|------------------------|\n')
    for f in sorted(files):
        st = classification_status(f)
        if st == 'KEEP':
            if f == 'solver_numerical.cpp': loc = 'src/numerical/main.cpp'
            elif f == 'solver_analytical.cpp': loc = 'src/analytical/main.cpp'
            elif f in ['solver_canonical.cpp', 'run_bisection.cpp', 'sweep_interval.cpp', 'stationary_finder.cpp', 'verify_potential.cpp']: loc = f'src/diagnostics/{f}'
            elif f.startswith('run_'): loc = f'python/runners/{f}'
            elif f.startswith('plot_'): loc = f'python/plotting/{f}'
            elif f.endswith('.md') and not f.startswith('paper/'): loc = f'docs/{f}'
            elif f.startswith('paper/'): loc = f
            elif f.startswith('results/'): loc = f.replace('results/', 'data/')
            else: loc = f'docs/{f}'
            out.write(f'| `{f}` | **KEEP** | `{loc}` |\n')
        else:
            out.write(f'| `{f}` | **{st}** | - |\n')

print("Created SMVAC_MIGRATION_PLAN.md")
