#!/usr/bin/env python3
"""
run_prompt_sensitivity.py

Runs the "Prompt Knowledge Sensitivity" experiment using 8 prompt variants (A–H)
across 3 SR models, 3 LLMs, and 3 physical experiments.

Total runs: 8 × 3 × 3 × 3 = 216 simulations
Outputs:
- Raw results: raw_prompt_results.csv
- Averaged metrics: prompt_sensitivity_avg.csv

NOTE: Before running, update SR_CHOICE and LLM_CHOICE with the best-performing models.
"""

import numpy as np
import csv
import pandas as pd
from sr_models import PySRModel, DEAPModel, GplearnModel
from llm_models import MistralLLM, LlamaLLM, FalconLLM
from ga_weight_optimizer import GAWeightOptimizer
from expression_tree import ExpressionTree
from run_main_claim import generate_synthetic_data, get_right_form

# Final model selections (adjust if needed)
SR_CHOICE = "pysr"      # pysr, gplearn, deap
LLM_CHOICE = "mistral"  # mistral, llama, falcon

SR_CLASSES = {"pysr": PySRModel, "gplearn": GplearnModel, "deap": DEAPModel}
LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}

# Data column descriptions (used in prompt B)
description_B = {
    "ball_drop": "Data columns: mass (kg), radius (m), air_resistance (kg/s), time (s), velocity (m/s), acceleration (m/s^2), height (m)",
    "shm": "Data columns: mass (kg), spring_const (N/m), amplitude (m), phase (rad), time (s), displacement (m), velocity (m/s), acceleration (m/s^2)",
    "wave": "Data columns: E0 (initial amplitude), alpha (damping factor), k (wave number), omega (angular frequency), x (m), t (s), E (field magnitude)"
}

# Prompt templates for A–H
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

def calculate_metrics(y_true, y_pred, ground_truth_expr, predicted_expr):
    """
    Computes regression metrics and structural distance.

    Returns:
        (mae, mse, r2, distance)
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred)**2))
    r2 = float(1 - mse / (np.var(y_true) + 1e-9))

    gt = ExpressionTree(); gt.build_from_string(ground_truth_expr)
    pred = ExpressionTree(); pred.build_from_string(predicted_expr)
    dist = float(pred.distance(gt))

    return mae, mse, r2, dist


def run_raw():
    """
    Runs all prompt × experiment × model combinations and logs raw results.
    """
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Experiment", "SR", "LLM", "MAE", "MSE", "R2", "Distance", "Equation"])

        for prompt_id, tmpl in prompt_templates.items():
            for sr_name, SRClass in SR_CLASSES.items():
                for llm_name, LLMClass in LLM_CLASSES.items():
                    for exp in experiments:
                        df = generate_synthetic_data(exp)
                        cols = {
                            "ball_drop": ['mass', 'radius', 'air_resistance', 'time', 'height'],
                            "shm": ['mass', 'spring_const', 'amplitude', 'phase', 'time'],
                            "wave": ['E0', 'alpha', 'k', 'omega', 'x', 't']
                        }[exp]
                        target = {
                            "ball_drop": "velocity",
                            "shm": "displacement",
                            "wave": "E"
                        }[exp]

                        X = df[cols].values
                        y = df[target].values
                        truth = get_right_form(exp)

                        B_text = description_B[exp]
                        C_text = prompt_templates["C"].format(exp=exp)
                        D_text = prompt_templates["D"].format(right_form=truth)
                        prompt = tmpl.format(B=B_text, C=C_text, D=D_text, exp=exp, right_form=truth)

                        sr = SRClass()
                        llm = LLMClass()

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

                        sr.fit(X, y)
                        expr = sr.get_best_equation() or ''
                        y_pred = sr.predict(X)
                        mae, mse, r2, dist = calculate_metrics(y, y_pred, truth, expr)

                        writer.writerow([prompt_id, exp, sr_name, llm_name, round(mae,4), round(mse,4), round(r2,4), round(dist,4), expr])


def run_avg():
    """
    Aggregates and averages the raw results.
    """
    df = pd.read_csv(raw_csv)
    grouped = df.groupby(["Prompt", "SR", "LLM"]).agg({
        "MAE": "mean", "MSE": "mean", "R2": "mean", "Distance": "mean",
        "Equation": lambda x: "|".join(x)
    }).reset_index()

    with open(avg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "SR", "LLM", "MAE", "MSE", "R2", "Distance", "Equations"])
        for _, row in grouped.iterrows():
            writer.writerow([
                row.Prompt, row.SR, row.LLM,
                round(row.MAE, 4), round(row.MSE, 4), round(row.R2, 4),
                round(row.Distance, 4), row.Equation
            ])


if __name__ == "__main__":
    run_raw()
    run_avg()
    print(f"[Prompt Sensitivity] Raw → {raw_csv}, Avg → {avg_csv}")
