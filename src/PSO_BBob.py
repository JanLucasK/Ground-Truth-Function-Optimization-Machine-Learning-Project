import torch
import bbobtorch
import random
import matplotlib.pyplot as plt
import numpy as np

# BBOB Function 01
fn = bbobtorch.create_f01(2, seed=42)  # two dimensions with seed 42

def bbob_function_01(x):
    return fn(torch.tensor(x))[0].item()

# Particle Swarm Optimization (PSO) Algorithm
def particle_swarm_optimization(dimensions, pop_size, generations):
    population = torch.tensor(np.random.uniform(-5, 5, (pop_size, dimensions)))
    velocities = torch.tensor(np.random.uniform(-1, 1, (pop_size, dimensions)))
    personal_best_positions = population.clone()
    personal_best_scores = [bbob_function_01(x.tolist()) for x in population]
    global_best_position = population[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5

    for _ in range(generations):
        for i in range(pop_size):
            velocities[i] = (inertia_weight * velocities[i] +
                            cognitive_weight * random.random() * (personal_best_positions[i] - population[i]) +
                            social_weight * random.random() * (global_best_position - population[i]))
            population[i] += velocities[i]
            score = bbob_function_01(population[i].tolist())
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

    return global_best_position

# PSO Parameters
dimensions = 2  # Dimension of the problem (2D)
pop_size = 50  # Population size
generations = 100  # Number of generations

# Run the PSO Algorithm
global_optimum = particle_swarm_optimization(dimensions, pop_size, generations)
global_optimum_value = bbob_function_01(global_optimum.tolist())

print(f'Global optimum found: {global_optimum}')
print(f'Function value at the optimum: {global_optimum_value}')

# Plot the function with the found optimum
x = torch.arange(-5, 5, 0.01, dtype=torch.float32)
xgrid, ygrid = np.meshgrid(x.numpy(), x.numpy())
results_grid = np.reshape([bbob_function_01([x, y]) for x, y in zip(xgrid.ravel(), ygrid.ravel())], xgrid.shape)

plt.figure(figsize=(6, 6))
plt.pcolormesh(xgrid, ygrid, results_grid, cmap='inferno', shading='auto')
plt.scatter(*global_optimum.tolist(), marker='x', c='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('BBOB Function 01 with PSO-found Optimum')
plt.colorbar()
plt.show()
