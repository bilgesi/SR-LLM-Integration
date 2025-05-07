#!/usr/bin/env python3
"""
run_main_claim.py

Runs the "Main Claim" experiment evaluating symbolic regression (SR) models
guided by LLMs over three physical simulation types:
    - Ball Drop
    - Simple Harmonic Motion (SHM)
    - Wave Propagation

Each run consists of:
    - 3 SR models × 3 LLMs × 3 experiments = 27 combinations
    - Iterates GA optimization until MAE improves < 0.1% or max 20 iterations
    - Outputs: main_claim_results.csv
"""

import numpy as np
import csv
from sr_models import PySRModel, DEAPModel, GplearnModel
from llm_models import MistralLLM, LlamaLLM, FalconLLM
from ga_weight_optimizer import GAWeightOptimizer
from expression_tree import ExpressionTree


def generate_synthetic_data(exp_type: str, n_samples: int = 500, noise_std: float = 0.01):
    """
    Generates synthetic time-series data for the given experiment type.

    Parameters:
        exp_type (str): 'ball_drop', 'shm', or 'wave'
        n_samples (int): Number of samples to generate
        noise_std (float): Relative standard deviation of Gaussian noise

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
    """
    g = 9.81
    m = np.random.uniform(0.1, 10.0, size=n_samples)
    r = np.random.uniform(0.01, 0.5, size=n_samples)
    h = np.random.uniform(1.0, 100.0, size=n_samples)
    air_res = np.random.uniform(0.0, 1.0, size=n_samples)
    t = np.random.uniform(0.0, np.sqrt(2 * h / g), size=n_samples)

    if exp_type == "ball_drop":
        v = np.sqrt(2 * g * h)
    elif exp_type == "shm":
        A, k, phi = 1.0, 1.0, 0.0
        v = A * np.cos(np.sqrt(k / m) * t + phi)
    elif exp_type == "wave":
        alpha, k_w, omega = 0.1, 1.0, 2 * np.pi
        x = np.zeros_like(t)
        v = np.exp(-alpha * t / 2) * np.cos(k_w * x - omega * t)
    else:
        raise ValueError(f"Unknown experiment: {exp_type}")

    y = v + np.random.normal(0, noise_std * np.std(v), size=n_samples)
    X = np.column_stack([m, r, h, air_res, t])
    return X, y


def get_right_form(exp_type: str) -> str:
    """
    Returns the canonical (ground-truth) formula for each experiment.

    Parameters:
        exp_type (str): Experiment name

    Returns:
        str: symbolic expression as string
    """
    if exp_type == "ball_drop":
        return "v(t) = sqrt(2 * g * h)"
    if exp_type == "shm":
        return "x(t) = A * cos(sqrt(k/m) * t + φ)"
    if exp_type == "wave":
        return "E0 * exp(-α * t / 2) * cos(k * x - ω * t)"
    raise ValueError(f"Unknown experiment type: {exp_type}")


# SR and LLM model mappings
SR_CLASSES = {"pysr": PySRModel, "gplearn": GplearnModel, "deap": DEAPModel}
LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}


def main():
    """
    Executes the full experiment suite:
    For each combination of experiment × SR model × LLM,
    runs GA optimization and logs evaluation metrics.
    """
    experiments = ["ball_drop", "shm", "wave"]
    output_csv = "main_claim_results.csv"
    header = ["Experiment", "SR", "LLM", "MAE", "MSE", "R2", "Distance", "Equation"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for exp in experiments:
            X, y = generate_synthetic_data(exp)
            truth = get_right_form(exp)

            for sr_name, SRClass in SR_CLASSES.items():
                for llm_name, LLMClass in LLM_CLASSES.items():
                    sr = SRClass()
                    llm = LLMClass()

                    ga = GAWeightOptimizer(
                        sr_model=sr,
                        llm_model=llm,
                        X=X,
                        y=y,
                        right_equation_str=truth,
                        experiment_type=exp,
                        sr_name=sr_name,
                        llm_name=llm_name,
                        pop_size=20,
                        generations=5,
                        mutation_rate=0.3,
                        results_csv=None,
                    )

                    # Iterative optimization until convergence
                    prev_mae = float("inf")
                    for _ in range(20):
                        ga.optimize()
                        sr.fit(X, y)
                        expr = sr.get_best_equation() or ""
                        y_pred = sr.predict(X)
                        mae = float(np.mean(np.abs(y - y_pred)))
                        if abs(prev_mae - mae) / (prev_mae + 1e-9) < 1e-3:
                            break
                        prev_mae = mae

                    # Final metrics
                    mse = float(np.mean((y - y_pred) ** 2))
                    r2 = 1 - mse / float(np.mean((y - np.mean(y)) ** 2))

                    gt = ExpressionTree()
                    gt.build_from_string(truth)
                    pred = ExpressionTree()
                    pred.build_from_string(expr)
                    dist = pred.distance(gt)

                    writer.writerow([
                        exp, sr_name, llm_name,
                        round(mae, 4), round(mse, 4), round(r2, 4),
                        round(dist, 4), expr
                    ])

    print(f"[Main Claim] Results saved to {output_csv}")


if __name__ == "__main__":
    main()
