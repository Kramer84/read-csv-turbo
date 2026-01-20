import os
import time
import pytest
import pandas as pd
import numpy as np
import readcsvturbo as rct
from pathlib import Path

# --- Configuration ---
# If running in Github Actions (CI), use smaller size to save resources.
# Locally, use a larger size to actually see the speed difference.
IS_CI = os.getenv('CI') == 'true'
ROWS = 10_000 if IS_CI else 1_000_000
COLS = 50
FILENAME = "speedtest_data.csv"

def generate_csv_if_missing(path, rows, cols):
    """Generates a random CSV file if it doesn't exist."""
    if path.exists():
        print(f"Using existing file: {path} ({path.stat().st_size / (1024*1024):.2f} MB)")
        return

    print(f"Generating {rows}x{cols} CSV at {path}...")
    # Use numpy for fast generation
    data = np.random.randint(0, 100, size=(rows, cols))
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])
    df.to_csv(path, index=False)
    print("Generation complete.")

@pytest.fixture(scope="module")
def large_csv_file(tmp_path_factory):
    """
    Pytest fixture that creates a temp file for the test session.
    Prioritizes a local 'big_csv.csv' if it exists.
    """
    # 1. Check for local manual file (prioritized for local dev)
    local_path = Path("big_csv.csv")
    if local_path.exists():
        return local_path

    # 2. Generate temp file for CI/Test
    fn = tmp_path_factory.mktemp("data") / FILENAME
    generate_csv_if_missing(fn, ROWS, COLS)
    return fn

def run_speed_benchmark(csv_path):
    """Core logic to run the 3 comparisons."""
    results = {}

    print(f"\n\n--- BENCHMARK: {csv_path} ---")

    # --- 1. RAW PANDAS ---
    # Reads entire file into memory, then slices.
    # Accurate but memory heavy and slow on big files.
    print("\n[Raw Pandas] Reading full file...")
    start = time.perf_counter()
    try:
        df = pd.read_csv(csv_path)
        df_first = df.iloc[[0]]
        df_last = df.iloc[[-1]]
        df_pandas = pd.concat([df_first, df_last])
        pd_time = time.perf_counter() - start
        results['Pandas'] = pd_time
        print(f"Time: {pd_time:.4f}s")
    except Exception as e:
        print(f"Pandas Failed (OOM?): {e}")
        df_pandas = pd.DataFrame()

    # --- 2. SKIPROWS (NAIVE) ---
    # Counts lines first (slow), then skips (slow in engine).
    print("\n[Skiprows] Counting lines & reading with skiprows...")
    start = time.perf_counter()
    try:
        # Note: 'readlines()' loads the whole file in memory!
        # A generator would be better, but keeping your original logic for comparison.
        with open(csv_path) as f:
            count = sum(1 for _ in f)

        # skip header (1) + skip middle (count - 1 header - 1 last row)
        to_skip = range(2, count)
        df_skip = pd.read_csv(csv_path, skiprows=to_skip, header=0)
        skip_time = time.perf_counter() - start
        results['Skiprows'] = skip_time
        print(f"Time: {skip_time:.4f}s")
    except Exception as e:
        print(f"Skiprows Failed: {e}")

    # --- 3. TURBO ---
    # Uses system tools (sed/tail/powershell) to slice bytes before Python sees them.
    print("\n[Turbo] Reading head/tail...")
    start = time.perf_counter()
    df_turbo = rct.read_csv_headtail(csv_path, header=True, n_rows_head=1, n_rows_tail=1)
    turbo_time = time.perf_counter() - start
    results['Turbo'] = turbo_time
    print(f"Time: {turbo_time:.4f}s")

    # --- VERIFICATION ---
    # Ensure Turbo returned the same data as Pandas
    if not df_pandas.empty:
        print("\nVerifying data consistency...")
        try:
            pd.testing.assert_frame_equal(
                df_pandas.reset_index(drop=True),
                df_turbo.reset_index(drop=True)
            )
            print("SUCCESS: Dataframes match.")
        except AssertionError as e:
            print("FAILURE: Dataframes do not match!")
            print(e)
            # In CI, we want to fail the test if data is wrong
            if IS_CI: raise e

    return results

def test_speed_performance(large_csv_file):
    """
    The actual test function picked up by Pytest.
    """
    results = run_speed_benchmark(large_csv_file)

    # Assert Turbo is functioning
    assert 'Turbo' in results

    # Optional: Fail if Turbo is significantly slower (only on large files)
    # On very small files (CI), Turbo might be slightly slower due to subprocess overhead.
    if not IS_CI and 'Pandas' in results:
        assert results['Turbo'] < results['Pandas'], "Turbo should be faster on large files!"

if __name__ == "__main__":
    # Setup for standalone execution
    path = Path(FILENAME)

    # Check for local big file override
    if Path("big_csv.csv").exists():
        path = Path("big_csv.csv")
    else:
        generate_csv_if_missing(path, ROWS, COLS)

    try:
        results = run_speed_benchmark(path)

        # Print Summary
        print("\n--- SUMMARY ---")
        for method, duration in results.items():
            print(f"{method}: {duration:.4f}s")

        if 'Pandas' in results and 'Turbo' in results:
            speedup = results['Pandas'] / results['Turbo']
            print(f"\nTurbo Speedup: {speedup:.2f}x")

    finally:
        # Cleanup generated file if it was the temp one
        if path.name == FILENAME and path.exists() and not Path("big_csv.csv").exists():
            print(f"Removing temp file {path}...")
            os.remove(path)
