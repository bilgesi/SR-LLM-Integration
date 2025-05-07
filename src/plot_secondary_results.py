#!/usr/bin/env python3
"""
plot_secondary_results.py

Generates secondary plots for all experiments:

1. Bubble charts (MAE vs R² colored by Distance) for:
   - Main Claim (`main_claim_results.csv`)
   - Prompt Sensitivity (`prompt_sensitivity_avg.csv`)
   - Data Sensitivity (`data_sensitivity_results.csv`)

2. Line plots of metrics across model interactions:
   A. Prompt Sensitivity: per prompt condition
   B. Data Sensitivity: per data fraction/noise value
   C. GA fitness logs (if `all_ga_logs.csv` exists)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output directory
os.makedirs('plots', exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Bubble Charts for experiment summaries
# ----------------------------------------------------------------------------

def bubble_chart(df, x, y, c, title, fname):
    """
    Generate a bubble chart (scatter plot with color-coded values).

    Args:
        df (pd.DataFrame): Input data
        x (str): Column name for X-axis
        y (str): Column name for Y-axis
        c (str): Column name for color encoding
        title (str): Title of the plot
        fname (str): Output filename (saved under 'plots/')
    """
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df[x], df[y], c=df[c], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(sc, label=c)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', fname))
    plt.close()

# Load and plot bubble charts
if os.path.exists('main_claim_results.csv'):
    df_mc = pd.read_csv('main_claim_results.csv')
    bubble_chart(df_mc, 'MAE', 'R2', 'Distance', 'Main Claim', 'bubble_main_claim.png')

if os.path.exists('prompt_sensitivity_avg.csv'):
    df_ps = pd.read_csv('prompt_sensitivity_avg.csv')
    bubble_chart(df_ps, 'MAE', 'R2', 'Distance', 'Prompt Sensitivity (avg)', 'bubble_prompt_sensitivity.png')

def preprocess_data_sens(df):
    """
    Combine 'mode' and 'value' into a single label for bubble chart plotting.
    """
    df['mode_value'] = df['mode'] + ' ' + df['value'].astype(str)
    return df

if os.path.exists('data_sensitivity_results.csv'):
    df_ds = pd.read_csv('data_sensitivity_results.csv')
    df_ds = preprocess_data_sens(df_ds)
    bubble_chart(df_ds, 'MAE', 'R2', 'Distance', 'Data Sensitivity', 'bubble_data_sensitivity.png')

# ----------------------------------------------------------------------------
# 2. Line Plots
# ----------------------------------------------------------------------------

def plot_prompt_sensitivity(raw_csv):
    """
    Line plots showing how metrics vary across prompts.

    3x3 grid of (experiment × SR), with lines per LLM.
    """
    raw = pd.read_csv(raw_csv)
    prompt_order = ['A','B','C','D','E','F','G','H']
    raw['Prompt'] = pd.Categorical(raw['Prompt'], categories=prompt_order, ordered=True)
    exps = ['ball_drop','shm','wave']
    srs = ['pysr','gplearn','deap']
    llms = ['mistral','llama','falcon']
    for metric in ['MAE','R2','Distance']:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
        for i, exp in enumerate(exps):
            for j, sr in enumerate(srs):
                ax = axes[i, j]
                for llm in llms:
                    df_sub = raw[(raw['Experiment']==exp)&(raw['SR']==sr)&(raw['LLM']==llm)].sort_values('Prompt')
                    ax.plot(df_sub['Prompt'], df_sub[metric], marker='o', label=llm)
                if i == 0: ax.set_title(f'SR={sr}')
                if j == 0: ax.set_ylabel(exp)
                if i == 2: ax.set_xlabel('Prompt')
                ax.grid(True)
                if i == 0 and j == 0: ax.legend()
        fig.suptitle(f'Prompt Sensitivity: {metric} vs Prompt', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'plots/ps_line_{metric.lower()}.png')
        plt.close()

if os.path.exists('raw_prompt_results.csv'):
    plot_prompt_sensitivity('raw_prompt_results.csv')

def plot_data_sensitivity(ds_csv):
    """
    Line plots for data sensitivity results by mode (fraction/noise).
    """
    df = pd.read_csv(ds_csv)
    exps = ['ball_drop','shm','wave']
    srs = ['pysr','gplearn','deap']
    llms = ['mistral','llama','falcon']
    for mode in ['fraction','noise']:
        df_mode = df[df['mode']==mode]
        for metric in ['MAE','R2','Distance']:
            fig, axes = plt.subplots(3, 3, figsize=(15,12), sharex=True)
            for i, exp in enumerate(exps):
                for j, sr in enumerate(srs):
                    ax = axes[i, j]
                    for llm in llms:
                        df_sub = df_mode[(df_mode['Experiment']==exp)&(df_mode['SR']==sr)&(df_mode['LLM']==llm)]
                        ax.plot(df_sub['value'], df_sub[metric], marker='o', label=llm)
                    if i == 0: ax.set_title(f'SR={sr}')
                    if j == 0: ax.set_ylabel(exp)
                    if i == 2: ax.set_xlabel(mode)
                    ax.grid(True)
                    if i == 0 and j == 0: ax.legend()
            fig.suptitle(f'Data Sensitivity ({mode}): {metric} vs Value', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'plots/ds_{mode}_line_{metric.lower()}.png')
            plt.close()

if os.path.exists('data_sensitivity_results.csv'):
    plot_data_sensitivity('data_sensitivity_results.csv')

def plot_ga_logs(log_csv):
    """
    Line plots of GA fitness scores (best and average) over generations.
    """
    df = pd.read_csv(log_csv)
    exps = df['experiment'].unique().tolist()
    srs = df['sr'].unique().tolist()
    llms = df['llm'].unique().tolist()
    for metric in ['best_fitness','avg_fitness']:
        fig, axes = plt.subplots(len(exps), len(srs), figsize=(15,12), sharex=True)
        for i, exp in enumerate(exps):
            for j, sr in enumerate(srs):
                ax = axes[i, j]
                for llm in llms:
                    df_sub = df[(df['experiment']==exp)&(df['sr']==sr)&(df['llm']==llm)]
                    ax.plot(df_sub['generation'], df_sub[metric], label=llm)
                if i == 0: ax.set_title(f'SR={sr}')
                if j == 0: ax.set_ylabel(exp)
                if i == len(exps)-1: ax.set_xlabel('generation')
                ax.grid(True)
                if i == 0 and j == 0: ax.legend()
        fig.suptitle(f'GA Logs: {metric} over Generations', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'plots/ga_logs_{metric}.png')
        plt.close()

if os.path.exists('all_ga_logs.csv'):
    plot_ga_logs('all_ga_logs.csv')
