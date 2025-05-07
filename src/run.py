#!/usr/bin/env python3
"""
run.py

Master script to execute all experiments in sequence:
    1. Main Claim (main_claim_results.csv)
    2. Prompt Sensitivity (raw + averaged)
    3. Data Sensitivity (raw + averaged)
    4. Plot Generation (bubble + line plots)

Each step logs to stdout and writes results into CSVs and /plots.
"""

import subprocess
import sys

def run_script(script_name):
    print(f"\n[RUNNING] {script_name}...\n" + "-"*60)
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed.\n")
        sys.exit(1)
    else:
        print(f"[DONE] {script_name}\n" + "-"*60)

if __name__ == "__main__":
    print("\n=== STARTING FULL EXPERIMENT PIPELINE ===\n")
    
    run_script("run_main_claim.py")
    run_script("run_prompt_sensitivity.py")
    run_script("run_data_sensitivity.py")
    run_script("plot_secondary_results.py")

    print("\n=== ALL TASKS COMPLETED SUCCESSFULLY ===\n")
