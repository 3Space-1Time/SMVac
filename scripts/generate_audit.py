import json
import os
import datetime

with open('inventory.json', 'r') as f:
    files = json.load(f)

# Categories
def categorize(f):
    path = f['path'].replace('\\', '/')
    name = f['file']
    ext = os.path.splitext(name)[1]
    
    if path.startswith('paper/'):
        if ext == '.tex' or ext == '.bib': return 'LaTeX manuscript'
        if ext == '.pdf': return 'Generated output'
        if 'figures/' in path: return 'Figures'
        return 'Temporary/debug files'
        
    if path.startswith('results/'):
        if ext == '.csv': return 'CSV outputs'
        if ext == '.png': return 'Figures'
        return 'Generated output'
        
    if path.startswith('archive/'):
        return 'Obsolete code'
        
    if path.startswith('src/'):
        if ext == '.cpp' or ext == '.py': return 'Older Source Code / Debug tools'
        if ext == '.exe': return 'Temporary/debug files'
        return 'Temporary/debug files'
        
    if ext == '.exe': return 'Executables'
    if ext == '.py':
        if 'plot_' in name: return 'Plotting scripts'
        if 'run_' in name: return 'Numerical integration wrappers'
        return 'Python utilities'
        
    if ext == '.cpp':
        if name in ['solver_analytical.cpp', 'solver_numerical.cpp', 'solver_numerical_dense.cpp', 'solver_analytical_dense.cpp']:
            return 'Core physics code'
        if name in ['solver_canonical.cpp', 'solver_instrumented.cpp', 'sweep_interval.cpp', 'run_bisection.cpp', 'find_interval.cpp', 'stationary_finder.cpp', 'verify_potential.cpp']:
            return 'Diagnostic solvers'
        return 'ODE solvers / Diagnostics'
        
    if ext == '.md': return 'Markdown reports'
    if ext in ['.log', '.txt', '.json']: return 'Temporary/debug files'
    
    return 'Other'

# Categorize all
grouped = {}
for f in files:
    cat = categorize(f)
    if cat not in grouped: grouped[cat] = []
    grouped[cat].append(f)

with open('PROJECT_AUDIT.md', 'w', encoding='utf-8') as out:
    out.write('# Project Audit: Threshold\n\n')
    
    out.write('## Part 1 — File Inventory\n\n')
    for cat, flist in grouped.items():
        out.write(f'### {cat}\n')
        out.write('| Filename | Path | Size | Last Modified | Description | Classification |\n')
        out.write('|----------|------|------|---------------|-------------|----------------|\n')
        for f in flist:
            p = f['path'].replace('\\', '/')
            sz = f['size']
            mt = f['mtime']
            # Basic heuristics
            if p.endswith('.cpp'): cls = 'Source code'
            elif p.endswith('.py'): cls = 'Source code'
            elif p.endswith('.exe'): cls = 'Executable'
            elif p.endswith('.md'): cls = 'Documentation'
            elif p.endswith('.csv'): cls = 'Generated output'
            elif p.endswith('.png'): cls = 'Plot'
            else: cls = 'Temporary/debug file'
            
            desc = "Core pipeline" if cat == "Core physics code" else ("Diagnostic" if "Diag" in cat else "Utility/Data")
            if p.endswith('.exe'): desc = "Compiled binary"
            
            out.write(f'| `{f["file"]}` | `{p}` | {sz} | {mt} | {desc} | {cls} |\n')
        out.write('\n')

    out.write('## Part 2 — Dependency Graph\n\n')
    out.write('''
### Core Pipeline
* `solver_numerical.cpp` -> `solver_numerical.exe`
* `solver_analytical.cpp` -> `solver_analytical.exe`
* `run_numerical.py` -> calls `solver_numerical.exe` -> generates `data/numerical_data.csv`
* `run_analytical.py` -> calls `solver_analytical.exe` -> generates `data/analytical_data.csv`
* `plot_numerical.py`, `plot_analytical.py`, `plot_overlay.py` -> reads `results/*.csv` -> generates `results/*.png`

### Diagnostic Pipeline
* `verify_potential.cpp` -> extracts exact RGE potential.
* `stationary_finder.cpp` -> locates EW minimum, barrier, high-field minimum.
* `sweep_interval.cpp` -> determines shooting behavior across the barrier/minimum sandwich.
* `run_bisection.cpp` -> computes exact converged canonical bounce.
* `plot_bisection.py`, `plot_traces.py` -> plots diagnostics.

### Obsolete / Archived
* All files in `archive/` and `src/sm_stability/` represent earlier prototyping and debug logic. They are no longer part of the primary generation pipeline.
''')

    out.write('\n## Part 3 — Physics Results Inventory\n\n')
    out.write('''| Result | Purpose | Generating Code | Outputs | Status | Publication? | GitHub? |
|--------|---------|-----------------|---------|--------|--------------|---------|
| Phase Diagram (Numerical) | Determine exact metastability boundary | `run_numerical.py` | `numerical_data.csv`, `numerical_stability_plot.png` | Validated | Yes | Yes |
| Phase Diagram (Analytical) | Baseline conformal approximation | `run_analytical.py` | `analytical_data.csv`, `analytical_stability_plot.png` | Validated | Yes | Yes |
| Disagreement Overlay | Highlight conformal vs exact difference | `plot_overlay.py` | `overlay_plot.png` | Validated | Yes | Yes |
| Sandwich Region Potential | Identify breakdown of conformal ansatz | `verify_potential.cpp`, `stationary_finder.cpp` | `POTENTIAL_STRUCTURE_REPORT.md` | Validated | Yes | Yes |
| Canonical Bounce Validation | Prove exact bounce convergence via ODE | `run_bisection.cpp`, `sweep_interval.cpp` | `bisection_converged.csv`, `bisection_plot.png` | Validated | Yes (appendix) | Yes |
''')

    out.write('\n## Part 4 — Repository Classification\n\n')
    out.write('''| Path Pattern | Classification | Reasoning |
|--------------|----------------|-----------|
| `solver_*.cpp` (root) | KEEP | Core production C++ code |
| `run_*.py` (root) | KEEP | Core parallel execution drivers |
| `plot_*.py` (root) | KEEP | Core plotting scripts |
| `paper/` | KEEP | LaTeX manuscript and associated figures |
| `*.exe` | DELETE | Compiled binaries should not be tracked |
| `*.log`, `*.txt`, `*.json` | DELETE | Temporary logs and data |
| `results/*.csv` | KEEP (LFS) / REGENERATE | Expensive results. Track via LFS or regenerate. |
| `results/*.png` | REGENERATE | Can be regenerated from CSV |
| `archive/` | ARCHIVE | Deprecated code, should be moved to a separate release or archival branch. |
| `src/sm_stability/` | ARCHIVE | Early prototyping code. |
| `*.md` | KEEP / ARCHIVE | AI reports. Should be distilled into GitHub docs or archived. |
''')

    out.write('\n## Part 5 — Proposed Research Repository (SMVac)\n\n')
    out.write('''### Proposed Directory Structure
```
SMVac/
├── README.md                 # Main overview and compilation instructions
├── LICENSE                   # Open source license (e.g., MIT)
├── docs/                     # Technical documentation, markdown reports
├── src/                      # C++ source code for bounce solvers
│   ├── numerical_solver/     # Exact numerical integration
│   ├── analytical_solver/    # Conformal FL approximation
│   └── diagnostics/          # Bisection, potential scanners, stationary finders
├── python/                   # Python wrappers, chunk runners, plotting
│   ├── runners/              # run_analytical.py, run_numerical.py
│   └── plotting/             # plot_overlay.py, plot_phase_diagram.py
├── data/                     # CSV outputs (cached via git LFS)
├── figures/                  # Generated PNGs
├── paper/                    # LaTeX manuscript
└── scripts/                  # Convenience shell/batch scripts for pipeline execution
```

### File Mapping
| Old Location | New Location | Action |
|--------------|--------------|--------|
| `solver_numerical.cpp` | `src/numerical_solver/main.cpp` | Move |
| `solver_analytical.cpp` | `src/analytical_solver/main.cpp` | Move |
| `run_numerical.py` | `python/runners/run_numerical.py` | Move |
| `plot_overlay.py` | `python/plotting/plot_overlay.py` | Move |
| `results/*.csv` | `data/` | Move |
| `results/*.png` | `figures/` | Move |
| `paper/` | `paper/` | Keep |
| `archive/` | - | Do not include |
| `src/sm_stability/` | - | Do not include |
| `*.exe` | - | Do not include |
| `*.md` (Reports) | `docs/` | Move |
''')

    out.write('\n## Part 6 — README Planning\n\n')
    out.write('''* **Repository Title:** SMVac: Exact Numerical Evaluation of Standard Model Vacuum Decay
* **Short Description:** A high-precision C++ and Python pipeline for computing the Standard Model electroweak vacuum instability boundary, rigorously avoiding the standard conformal high-field approximation.
* **Main Scientific Contributions:** 
  1. Exact integration of the O(4)-symmetric bounce action.
  2. Proof of failure of the conformal Fubini-Lipatov ansatz near the metastability boundary.
  3. Analysis of the "sandwiched" instability interval with positive boundary potential slopes.
* **Compilation Instructions:** Uses standard C++17 with OpenMP. `g++ -std=c++17 -O3 -fopenmp src/numerical_solver/main.cpp -o numerical_solver.exe`.
* **Reproducibility Workflow:** `python python/runners/run_numerical.py` to generate the 1M-point grid, followed by `python python/plotting/plot_overlay.py` to generate manuscript figures.
* **Examples:** Running a single benchmark point (e.g., $M_h=5, M_t=105$) using the diagnostics suite.
* **Expected Outputs:** Phase diagrams (stable, metastable, unstable) mapping the $M_h$-$M_t$ plane.
* **Future Work:** Generalization to non-standard Higgs sectors, BSM dimension-6 operators, and thermal bounce transitions.
''')

print("Created PROJECT_AUDIT.md")
