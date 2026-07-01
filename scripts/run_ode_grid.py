import subprocess
import concurrent.futures
import pandas as pd
import glob
import os
import time

TOTAL_POINTS = 63001
NUM_PROCESSES = 12

chunk_size = TOTAL_POINTS // NUM_PROCESSES + 1

def run_chunk(start, end):
    print(f"Running chunk {start} to {end}...")
    subprocess.run(["solver_ode_grid.exe", str(start), str(end)])
    print(f"Finished chunk {start} to {end}.")

if __name__ == "__main__":
    t0 = time.time()
    chunks = []
    for i in range(NUM_PROCESSES):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, TOTAL_POINTS)
        chunks.append((start, end))

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = [executor.submit(run_chunk, s, e) for s, e in chunks]
        concurrent.futures.wait(futures)

    print(f"All chunks completed in {time.time()-t0:.1f} seconds. Combining CSV files...")

    all_files = glob.glob("results/ode_grid_data_chunk_*.csv")
    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))
        
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv("data/ode_grid_data.csv", index=False)

    print("Combined file saved. Cleaning up chunks...")
    for f in all_files:
        os.remove(f)
        
    print("Done!")
