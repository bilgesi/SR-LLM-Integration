# Pre-trained Large Language Models Based Knowledge Integration in Physical Symbolic Regression

## Abstract
Symbolic regression (SR) provides a powerful means for discovering interpretable equations from data, 
especially in physics. In this research, we present a modular open-source framework that 
integrates SR with large-scale language models (LLMs) to improve the credibility and stability 
of the discovered physics equations. Our approach systematically evaluates a combination of 
symbolic regression methods (PySR, DEAP-GP, gplearn) and pre-trained LLMs (Mistral 7B, Llama 2 7B, Falcon 7B) 
through a genetic algorithm-based optimization process that strikes a balance between prediction accuracy, 
expressive simplicity, and scientific plausibility of LLM scores.

We evaluated the system in three simulated physical systems. Extensive cue-sensitivity and 
data-sensitivity experiments were also conducted. The results show that combining 
SR with LLM-induced evaluation reveals more accurate, dimensionally consistent, 
and semantically richer expressions, even in noisy environments.


## Table of contents
1. [Code usage](#code_usage)
2. [Experimental design](#experimental_design)
3. [Data](#data_preparation)
4. [Dependencies](#dependencies)
5. [Contributing](#contributing)
6. [Bug Reports](#bug_reports)
7. [Contact](#contact)

<a name="code_usage"/>

## Code usage
### Run the experiments shown in the paper:
1. Clone the repo 
2. Install the `requirements.txt` file.
3. Login to `Hugging Face` with token.
4. run the project from the `run.py` file. 

<a name="experimental_design"/>

## Experimental design
Our symbolic regression (SR) and language model (LLM) pipeline is composed of four tightly integrated components: 
1. **Data Generator:** For each physics experiment type (free-fall, simple harmonic motion, wave propagation), we simulated 500 samples by sampling physical parameters (e.g., mass, height, spring constants) and generated noise-enhanced ground-truth output. This forms the input to the SR + LLM processing (run_main_claim.py).
2. **Symbolic Regression Engines (SR):** We support multiple symbolic regression models—including PySR, DEAP and gplearn—each configured with 100 population size and 8 evolution iterations. These models independently evolve mathematical expressions that best fit the data using genetic programming or evolutionary search (sr_models.py).
3. **LLM-Guided Fitness Optimization:** We incorporate large language models (Mistral, Llama, Falcon) into the SR loop via a genetic algorithm that learns optimal weights for three fitness components: numerical accuracy (MSE), plausibility score (1 − sim), and expression complexity. Each candidate expression is scored by both SR and LLM, and a composite loss guides evolution (ga_weight_optimizer.py).
4. **Prompt Sensitivity Modules:** We used eight controlled prompt variants (A-H) to evaluate how SR changes under different LLM natural language wordings (run_prompt_sensitivity.py).
5. **Data Sensitivity Modules:** We evaluate model robustness under changes in sample size (50%–150%) and noise level (1%–5%) to ensure generalization and resilience (run_data_sensitivity.py).

All components are orchestrated by run.py, which executes the entire experiment and saves both raw results and averaged metrics. Plots summarizing the main claim, prompt impact, and data robustness are automatically generated via plot_secondary_results.py.

![structure](https://github.com/bilgesi/SR-LLM-Integration/blob/76a0f2fa4dd813acdbff644cd7ffc6a1553a2abb/src/images/srllm.png)

<a name="data_preparation"/>

## Data preparation
All experiments in this framework use class-generated physics datasets that simulate three classical physics systems: free fall, simple harmonic motion, and damped wave propagation. For each case, a dataset of 500 samples is created by randomly sampling physically meaningful parameters, see run_main_claim.py for details. Custom generation of data for other physics experiments is also supported, but the data files should be csv files.

The results of running the experiment will be saved in a directory named “results”, which will generate the final_results.csv file and a visualization view.

<a name="dependencies"/>

## Dependencies 
1. pandas 
2. numpy 
3. matplotlib 
4. seaborn 
5. scikit-learn 
6. pysr
7. deap
8. gplearn 
9. transformers 
10. torch 
11. bitsandbytes
12. sympy

These dependencies are the basic functional dependencies of the project, please refer to requirements.txt for detailed dependencies.

## How to cite

If you use this work, please cite:

```bibtex
@article{taskin2026knowledge_integration_pisr,
  title   = {Knowledge integration for physics-informed symbolic regression using pre-trained large language models},
  author  = {Taskin, Bilge and Xie, Wenxiong and Lazebnik, Teddy},
  journal = {Scientific Reports},
  volume  = {16},
  pages   = {1614},
  year    = {2026},
  doi     = {10.1038/s41598-026-35327-6},
  url     = {https://doi.org/10.1038/s41598-026-35327-6}
}


<a name="contributing"/>

## Contributing
We would love you to contribute to this project, pull requests are very welcome! Please send us an email with your suggestions or requests...

<a name="bug_reports"/>

## Bug Reports
Report [here]("https://github.com/bilgesi/SR-LLM-Integration/tree/ec195859ded6aa81d003798d03856f26a0a3f23b/src/issues"). Guaranteed reply as fast as we can :)

<a name="contact"/>

## Contact
* Bilge Taskin - [email](tabi23yu@student.ju.se)
* Wenxiong Xie - [email](xiwe23hu@student.ju.se)
