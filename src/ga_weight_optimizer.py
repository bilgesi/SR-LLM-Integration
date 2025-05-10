# ga_weight_optimizer.py

import numpy as np
import random
import os
from expression_tree import ExpressionTree
from src.constants import FITNESS_FAIL_SCORE, EARLY_STOP_TOLERANCE, STALE_GEN_LIMIT, MUTATION_STD_DEV, EPSILON


class GAWeightOptimizer:
    """
    Genetic algorithm for optimizing the weights (w1, w2, w3) used in symbolic regression fitness.

    Fitness function:
        fitness = w1 * MSE + w2 * (1 - LLM_score) + w3 * complexity

    Parameters:
    ----------
    sr_model : object
        Symbolic regression model implementing fit(), predict(), get_best_equation().
    llm_model : object
        LLM scoring model implementing evaluate_equation().
    X, y : ndarray
        Training data.
    right_equation_str : str
        The ground truth equation (used for computing distance or complexity).
    experiment_type, sr_name, llm_name : str
        Metadata for experiment logging.
    pop_size : int
        Population size.
    generations : int
        Max generations for GA.
    mutation_rate : float
        Probability of mutation.
    results_csv : str or None
        Path to CSV for logging GA progress.
    prompt_text : str
        Context string used by the LLM model.
    """

    def __init__(
        self,
        sr_model,
        llm_model,
        X, y,
        right_equation_str,
        experiment_type,
        sr_name,
        llm_name,
        pop_size=20,
        generations=5,
        mutation_rate=0.3,
        results_csv="all_ga_logs.csv",
        prompt_text="",
    ):
        self.sr_model = sr_model
        self.llm_model = llm_model
        self.X, self.y = X, y
        self.right_tree = ExpressionTree()
        self.right_tree.build_from_string(right_equation_str)

        self.exp = experiment_type
        self.sr_name = sr_name
        self.llm_name = llm_name

        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.results_csv = results_csv
        self.prompt_text = prompt_text

    def _loss_function(self, weights):
        """
        Compute weighted fitness using symbolic regression + LLM + complexity.
        """
        if len(weights) != 3 or not np.isclose(weights.sum(), 1.0, atol=EPSILON):
            return FITNESS_FAIL_SCORE

        w1, w2, w3 = weights
        try:
            self.sr_model.fit(self.X, self.y)
            eq_str = self.sr_model.get_best_equation() or ""
            if not eq_str:
                return FITNESS_FAIL_SCORE

            y_pred = self.sr_model.predict(self.X)
            mse = float(np.mean((self.y - y_pred) ** 2))

            llm_score = self.llm_model.evaluate_equation(eq_str, context=self.prompt_text)

            try:
                t = ExpressionTree()
                t.build_from_string(eq_str)
                complexity = float(t.compute_complexity())
            except Exception:
                complexity = float('inf')

            return w1 * mse + w2 * (1 - llm_score) + w3 * complexity
        except Exception:
            return FITNESS_FAIL_SCORE

    def optimize(self):
        """
        Run the genetic algorithm and return the best weight vector and its fitness.
        """
        pop = [np.random.dirichlet([3, 2, 1]) for _ in range(self.pop_size)]
        log_to_file = bool(self.results_csv)

        if log_to_file:
            new_file = not os.path.exists(self.results_csv)
            with open(self.results_csv, 'a') as f:
                if new_file:
                    f.write('experiment,sr,llm,generation,best_fitness,avg_fitness\n')

        stale_gens = 0
        prev_best = float('inf')

        for gen in range(self.generations):
            fitness_vals = np.array([self._loss_function(ind) for ind in pop])
            order = fitness_vals.argsort()
            best_fit = float(fitness_vals[order[0]])
            avg_fit = float(fitness_vals.mean())

            if log_to_file:
                with open(self.results_csv, 'a') as f:
                    f.write(f"{self.exp},{self.sr_name},{self.llm_name},{gen},{best_fit},{avg_fit}\n")

            improvement = (prev_best - best_fit) / (abs(prev_best) + EPSILON)
            stale_gens = stale_gens + 1 if improvement < EARLY_STOP_TOLERANCE else 0
            prev_best = best_fit

            if stale_gens >= STALE_GEN_LIMIT:
                print(f"[GA] Early stopping at generation {gen} – fitness plateaued.")
                break

            elites = [pop[i] for i in order[:self.pop_size // 2]]
            offspring = []
            for _ in range(self.pop_size // 2):
                p1, p2 = random.sample(elites, 2)
                child = (p1 + p2) / 2
                if random.random() < self.mutation_rate:
                    child += np.random.normal(0, MUTATION_STD_DEV, size=3)
                child = np.clip(child, 0, 1)
                child = child / child.sum() if child.sum() > 0 else np.array([1, 0, 0])
                offspring.append(child)
            pop = elites + offspring

        fitness_vals = np.array([self._loss_function(ind) for ind in pop])
        best_idx = int(fitness_vals.argmin())
        return pop[best_idx], float(fitness_vals[best_idx])
