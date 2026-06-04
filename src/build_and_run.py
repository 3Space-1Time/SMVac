import os
import subprocess
import sys

def run_command(cmd, cwd=None, env=None):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except subprocess.CalledProcessError:
        print(f"Error executing command: {' '.join(cmd)}")
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sm_dir = os.path.join(base_dir, "sm_stability")
    plot_dir = os.path.join(base_dir, "plotting")
    
    # Check if simplebounce exists
    sb_dir = os.path.abspath(os.path.join(base_dir, "..", "external", "SimpleBounce"))
    if not os.path.exists(sb_dir):
        print("SimpleBounce not found, cloning...")
        ext_dir = os.path.abspath(os.path.join(base_dir, "..", "external"))
        os.makedirs(ext_dir, exist_ok=True)
        run_command(["git", "clone", "https://github.com/rsato64/SimpleBounce.git", sb_dir])

    # Verify portable compiler exists
    compiler_path = os.path.abspath(os.path.join(base_dir, "..", "external", "mingw64", "bin", "g++.exe"))
    if not os.path.exists(compiler_path):
        print(f"Warning: Portable compiler not found at {compiler_path}. Falling back to system g++.")
        compiler_path = "g++"

    # Compile the code
    print("--- Compiling C++ Solver with OpenMP ---")
    compile_cmd = [
        compiler_path, "-std=c++17", "-O3", "-fopenmp",
        "-I", sb_dir,
        "-o", "solver.exe",
        "bounce_action_solver.cpp",
        os.path.join(sb_dir, "simplebounce.cc")
    ]
    run_command(compile_cmd, cwd=sm_dir)

    # Run the simulation with updated PATH for DLLs
    print("--- Running Simulation ---")
    run_cmd = [os.path.abspath(os.path.join(sm_dir, "solver.exe"))]
    
    env = os.environ.copy()
    if os.path.exists(os.path.dirname(compiler_path)):
        env["PATH"] = os.path.dirname(compiler_path) + os.pathsep + env.get("PATH", "")
    
    run_command(run_cmd, cwd=sm_dir, env=env)

    # Plot
    print("--- Plotting ---")
    run_command(["python", "plot_phase_diagram.py"], cwd=plot_dir)

if __name__ == "__main__":
    main()
