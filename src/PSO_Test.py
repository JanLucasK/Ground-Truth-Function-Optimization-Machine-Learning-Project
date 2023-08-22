import numpy as np
import torch
import matplotlib.pyplot as plt

# Cost function used by the model
def cost_function(x, model):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    output = model(x_tensor)
    cost = -output.item()  # Convert output to cost
    return cost

# PSO step: Calculation of new position and velocity
def pso_step(positions, velocities, best_positions, global_best_position, c1, c2, w):
    new_velocities = w * velocities + c1 * np.random.rand() * (best_positions - positions) + c2 * np.random.rand() * (global_best_position - positions)
    new_positions = positions + new_velocities
    return new_positions, new_velocities

# PSO algorithm
def pso(cost_function, model, num_particles=50, num_dimensions=2, max_iterations=100, w=0.5, c1=1.5, c2=1.5):
    particles_position = np.random.rand(num_particles, num_dimensions)
    particles_velocity = np.random.rand(num_particles, num_dimensions)
    
    particles_best_position = particles_position.copy()
    particles_best_cost = np.zeros(num_particles) + np.inf
    
    global_best_position = particles_position[0].copy()
    global_best_cost = np.inf
    
    # Store particle trajectories
    particle_trajectories = [[] for _ in range(num_particles)]
    for i in range(num_particles):
        particle_trajectories[i].append(particles_position[i].tolist())
    
    for iteration in range(max_iterations):
        for i in range(num_particles):
            cost = cost_function(particles_position[i], model)
            
            if cost < particles_best_cost[i]:
                particles_best_cost[i] = cost
                particles_best_position[i] = particles_position[i]
            
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = particles_position[i]
                
            # Update trajectories
            particle_trajectories[i].append(particles_position[i].tolist())
                
        particles_position, particles_velocity = pso_step(
            particles_position, particles_velocity,
            particles_best_position, global_best_position,
            c1, c2, w
        )
        
    return global_best_position, global_best_cost, particle_trajectories

# Load the model
loaded_model = torch.load('models/training_v1_f01_2.pth')
loaded_model = loaded_model.eval()

# Start the PSO algorithm
global_best_position, global_best_cost, particle_trajectories = pso(cost_function, loaded_model)

print("Global optimum found at:", global_best_position)
print("Cost at global optimum:", global_best_cost)

# Plot the function with the found optimum and particle trajectories
x = torch.arange(-5, 5, 0.01, dtype=torch.float32)
xgrid, ygrid = np.meshgrid(x.numpy(), x.numpy())
results_grid = np.reshape([cost_function([x, y], loaded_model) for x, y in zip(xgrid.ravel(), ygrid.ravel())], xgrid.shape)

plt.figure(figsize=(8, 6))
plt.pcolormesh(xgrid, ygrid, results_grid, cmap='inferno', shading='auto')
for traj in particle_trajectories:
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], marker='', linewidth=1, color='white', alpha=0.5)
plt.scatter(*global_best_position.tolist(), marker='x', c='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Function 01 with PSO-found Optimum and Trajectories')
plt.colorbar()
plt.show()
