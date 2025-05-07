"""
sr_models.py

Provides symbolic regression model wrappers for use in LLM-guided optimization.
Supported:
- PySR
- DEAP (Genetic Programming)
- gplearn
- Dummy (baseline for debugging)
"""

import numpy as np
import sympy as sp

# PySR
try:
    from pysr import PySRRegressor
except ImportError:
    print("Warning: PySR not installed. Try `pip install pysr`.")

# DEAP
try:
    from deap import base, creator, tools, gp, algorithms
    import operator
    import random
except ImportError:
    print("Warning: DEAP not installed. Try `pip install deap`.")

# gplearn
try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:
    print("Warning: gplearn not installed. Try `pip install gplearn`.")


class AbstractSR:
    """
    Abstract base class for all symbolic regression wrappers.
    Must implement: fit(), predict(), get_best_equation().
    """
    def __init__(self):
        self.best_equation_ = None
        self.best_model_ = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def get_best_equation(self):
        return self.best_equation_


class PySRModel(AbstractSR):
    """
    Wrapper for the PySR symbolic regression library.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        self.model = PySRRegressor(
            niterations=8,
            population_size=100,
            elementwise_loss="HuberLoss()",
            complexity_of_operators={"^": 4},
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "log", "sin", "cos"],
            constraints={"^": (-1, 1)},
            extra_sympy_mappings={"pow": sp.Pow},
            verbosity=0,
            **self.kwargs
        )
        self.model.fit(X, y)
        best_equations_df = self.model.equations_.sort_values(by="loss")
        self.best_equation_ = str(best_equations_df.iloc[0]["equation"])
        self.best_model_ = self.model

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        return self.model.predict(X)


class DEAPModel(AbstractSR):
    """
    Wrapper for symbolic regression using DEAP (genetic programming).
    """
    def __init__(self, pop_size=100, gens=8, **kwargs):
        super().__init__()
        self.pop_size = pop_size
        self.gens = gens
        self.kwargs = kwargs

    def fit(self, X, y):
        import functools

        def protected_div(left, right):
            return 1.0 if abs(right) < 1e-9 else left / right

        if len(X.shape) > 1 and X.shape[1] > 1:
            X = X[:, 0]

        pset = gp.PrimitiveSet("MAIN", 1)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addEphemeralConstant("rand101", functools.partial(np.random.randint, -1, 2))

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def eval_symb_reg(individual, X_, y_):
            func = toolbox.compile(expr=individual)
            preds = [func(xi) for xi in X_]
            return (np.mean((preds - y_) ** 2),)

        toolbox.register("evaluate", eval_symb_reg, X_=X, y_=y)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=8))
        toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=8))

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.6, mutpb=0.3,
            ngen=self.gens,
            stats=stats,
            halloffame=hof,
            verbose=False
        )

        self.toolbox = toolbox
        self.best_ind = hof[0]
        self.best_equation_ = str(self.best_ind)

    def predict(self, X):
        if len(X.shape) > 1 and X.shape[1] > 1:
            X = X[:, 0]
        func = self.toolbox.compile(expr=self.best_ind)
        return np.array([func(xi) for xi in X])


class GplearnModel(AbstractSR):
    """
    Wrapper for gplearn SymbolicRegressor.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.model = SymbolicRegressor(
            population_size=100,
            generations=8,
            function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
            parsimony_coefficient=5e5,
            stopping_criteria=5e4,
            verbose=0,
            **self.kwargs
        )
        self.model.fit(X, y)
        self.best_equation_ = str(self.model._program)
        self.best_model_ = self.model

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.model.predict(X)


class DummySR(AbstractSR):
    """
    Dummy symbolic regression model for fast testing/debugging.
    Always returns zero as prediction and '0' as equation.
    """
    def __init__(self):
        super().__init__()
        self.best_equation_ = "0"

    def fit(self, X, y):
        pass  # No fitting

    def predict(self, X):
        return np.zeros(len(X))

    def get_best_equation(self):
        return self.best_equation_
