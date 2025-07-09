from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt

def black_box_function(x):
    return np.sin(3 * x) + 0.5 * np.cos(5 * x)

pbounds = {"x": (0, 2 * np.pi)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=20
)

# Plot results
X_grid = np.linspace(pbounds["x"][0], pbounds["x"][1], 1000).reshape(-1, 1)
y_true = black_box_function(X_grid)

# Get all sampled points and values
sampled_x = np.array([res["params"]["x"] for res in optimizer.res])
sampled_y = np.array([res["target"] for res in optimizer.res])

plt.figure(figsize=(10, 6))
plt.plot(X_grid, y_true, 'r--', label='True function')
plt.plot(sampled_x, sampled_y, 'kx', label='Sampled points')
plt.title('Bayesian Optimization with Probability of Improvement (bayes_opt)')
plt.legend()
plt.show()

print("Best result:", optimizer.max)







