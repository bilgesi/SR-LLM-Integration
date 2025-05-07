#!/usr/bin/env python3
"""
run_prompt_sensitivity.py

Runs the "Prompt Knowledge Sensitivity" experiment with 8 prompt variations (A–H).
Each combination runs across 3 experiments × 3 SR models × 3 LLMs = 72 combinations total.

Outputs:
- Raw results → raw_prompt_results.csv
- Aggregated (averaged) metrics → prompt_sensitivity_avg.csv
"""

import numpy as np
import csv
import pandas as pd
from sr_models import PySRModel, DEAPModel, GplearnModel
from llm_models import MistralLLM, LlamaLLM, FalconLLM
from ga_weight_optimizer import GAWeightOptimizer
from run_main_claim import generate_synthetic_data, get_right_form
from expression_tree import ExpressionTree

# Mappings
SR_CLASSES = {"pysr": PySRModel, "gplearn": GplearnModel, "deap": DEAPModel}
LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}

# Context descriptions for each experiment (used in prompts)
description_B = {
    "ball_drop": (
        "Data columns: mass (kg), radius (m), air_resistance (kg/s), time (s), "
        "velocity (m/s), acceleration (m/s^2), height (m)"
    ),
    "shm": (
        "Data columns: mass (kg), spring_const (N/m), amplitude (m), phase (rad), "
        "time (s), displacement (m), velocity (m/s), acceleration (m/s^2)"
    ),
    "wave": (
        "Data columns: E0 (initial amplitude), alpha (damping factor), k (wave number), "
        "omega (angular frequency), x (position, m), t (time), E (field magnitude)"
    ),
}

# Prompt templates for sensitivity conditions A–H
prompt_templates = {
    "A": "",
    "B": "{B}",
    "C": "Experiment description: simulate {exp} dynamics with relevant physical context.",
    "D": "By the way, the right formula is {right_form}",
    "E": "{B}\n{C}",
    "F": "{B}\n{D}",
    "G": "{C}\n{D}",
    "H": "{B}\n{C}\n{D}",
}

experiments = ["ball_drop", "shm", "wave"]
raw_csv = "raw_prompt_results.csv"
avg_csv = "prompt_sensitivity_avg.csv"

def run_raw():
    """
    Runs all combinations of experiments × prompts × models and logs raw metrics.
    """
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Experiment", "SR", "LLM", "MAE", "MSE", "R2", "Distance", "Equation"])

        for prompt_id, tmpl in prompt_templates.items():
            for sr_name, SRClass in SR_CLASSES.items():
                for llm_name, LLMClass in LLM_CLASSES.items():
                    for exp in experiments:
                        X, y = generate_synthetic_data(exp)
                        truth = get_right_form(exp)

                        # Format the prompt using template
                        prompt = tmpl.format(
                            B=description_B[exp],
                            C=prompt_templates["C"].format(exp=exp),
                            D=prompt_templates["D"].format(right_form=truth),
                            exp=exp,
                            right_form=truth
                        )

                        # Initialize models
                        sr = SRClass()
                        llm = LLMClass()

                        # GA optimization
                        ga = GAWeightOptimizer(
                            sr_model=sr,
                            llm_model=llm,
                            X=X, y=y,
                            right_equation_str=truth,
                            experiment_type=exp,
                            sr_name=sr_name,
                            llm_name=llm_name,
                            pop_size=8,
                            generations=3,
                            mutation_rate=0.5,
                            results_csv=None,
                            prompt_text=prompt
                        )
                        ga.optimize()

                        # Final prediction
                        sr.fit(X, y)
                        expr = sr.get_best_equation() or ''
                        y_pred = sr.predict(X)

                        # Compute metrics
                        mae = float(np.mean(np.abs(y - y_pred)))
                        mse = float(np.mean((y - y_pred)**2))
                        r2 = float(1 - mse / (np.var(y) + 1e-9))

                        gt = ExpressionTree(); gt.build_from_string(truth)
                        pred = ExpressionTree(); pred.build_from_string(expr)
                        dist = float(pred.distance(gt))

                        # Log result
                        writer.writerow([
                            prompt_id, exp, sr_name, llm_name,
                            round(mae, 4), round(mse, 4), round(r2, 4),
                            round(dist, 4), expr
                        ])

def run_avg():
    """
    Aggregates raw results and computes average metrics per Prompt × SR × LLM.
    """
    df = pd.read_csv(raw_csv)
    grouped = df.groupby(["Prompt", "SR", "LLM"], as_index=False).agg({
        "MAE": "mean", "MSE": "mean", "R2": "mean", "Distance": "mean",
        "Equation": lambda x: "|".join(x)
    })
    for col in ["MAE", "MSE", "R2", "Distance"]:
        grouped[col] = grouped[col].round(4)
    grouped.to_csv(avg_csv, index=False)

if __name__ == "__main__":
    run_raw()
    run_avg()
    print(f"[Prompt Sensitivity] Raw → {raw_csv}, Avg → {avg_csv}")
