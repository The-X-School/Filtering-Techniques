from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import numpy as np

def bayop(black_box, pbounds, iter_points, iters):
    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=pbounds,
        random_state=42,
    )

    # Specify acquisition function: 'ei' = Expected Improvement
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    optimizer.maximize(
        init_points=iter_points,
        n_iter=iters,
        acq=utility.kind,  # this is equivalent to kind='ei'
    )

    # Print max result
    print("Max result:")
    print("Target (objective value):", optimizer.max['target'])
    print("Parameters (input values):", optimizer.max['params'])

    return optimizer.max
