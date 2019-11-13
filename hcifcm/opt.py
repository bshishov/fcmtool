import numpy as np
from hcifcm.fcm import FCM, simulate


def hebbian_learning(concept_values: np.ndarray,
                     edge_matrix: np.ndarray,
                     activation_fn: callable,
                     max_iterations: int = 1000,
                     neu: float = 0.001,
                     tolerance: float=1e-6):
    delta_w = np.zeros_like(edge_matrix)
    for iteration in range(max_iterations):
        # Weights update
        for i in range(edge_matrix.shape[0]):
            for j in range(edge_matrix.shape[1]):
                if np.abs(edge_matrix[j, i]) > tolerance:
                    delta_w[j, i] = concept_values[j] * (concept_values[i] - np.sign(edge_matrix[j, i])
                                                         * concept_values[j] * edge_matrix[j, i])
        edge_matrix = np.clip(edge_matrix + neu * delta_w, -1.0, 1.0)

        # Run simulation
        concept_values = simulate(concept_values, edge_matrix, activation_fn)
    return edge_matrix


def pso(fitness_fn: callable,
        iterations: int=1000,
        num_weights: int = 20,
        num_particles: int = 10,
        target_fitness: float=float('inf'),
        target_tolerance: float=1e-4,
        weight_min: [float, np.ndarray, list]=-1,
        weight_max: [float, np.ndarray, list]=1,
        inertia: float = 0.5,
        personal_importance: float = 0.8,
        social_importance: float = 0.9):
    particle_positions = weight_min + np.random.random((num_particles, num_weights)) * (weight_max - weight_min)
    particle_velocities = np.zeros_like(particle_positions)
    particle_best_position = particle_positions.copy()
    particle_best_fitness = np.full((num_particles,), float('-inf'))
    best_position = np.zeros(num_weights)
    best_fitness = float('-inf')

    for iteration in range(iterations):
        # Update best
        for i in range(num_particles):
            candidate_fitness = fitness_fn(particle_positions[i])
            if candidate_fitness > particle_best_fitness[i]:
                particle_best_fitness[i] = candidate_fitness
                particle_best_position[i] = particle_positions[i]
            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_position = particle_positions[i]

        if np.abs(best_fitness - target_fitness) <= target_tolerance:
            return best_position

        # Update velocities and move particles
        particle_velocities = inertia * particle_velocities + \
                              personal_importance * np.random.random((num_particles, 1)) * \
                              (particle_best_position - particle_positions) + \
                              social_importance * np.random.random() * \
                              (best_position - particle_positions)
        particle_positions = np.clip(particle_positions + particle_velocities, weight_min, weight_max)
    return best_position


def fcm_optimize(fcm: FCM,
                 concept_min,
                 concept_max):
    def _loss(c, c_min, c_max):
        return
    pass


class FcmPso(object):
    def __init__(self, fcm: FCM, concept_min=-1, concept_max=1):
        self.fcm = fcm
        self.concepts = fcm.get_concepts().copy()
        self.edge_matrix_shape = (len(self.concepts), len(self.concepts))
        self.num_weights = len(self.concepts) ** 2
        self.concept_min = concept_min
        self.concept_max = concept_max

    def fitness(self, weights):
        w = np.reshape(weights, self.edge_matrix_shape)
        c_new = simulate(self.concepts, w, self.fcm.activation)
        return np.sum(np.square(c_new - self.concepts))

    def optimize(self, *args, **kwargs):
        new_weights = pso(self.fitness,
                          num_weights=self.num_weights,
                          weight_min=self.concept_min,
                          weight_max=self.concept_max,
                          *args, **kwargs)
        self.fcm.set_weight_matrix(new_weights.reshape(self.edge_matrix_shape))


def _fitness(args):
    return -(args[0] ** 2 + args[1] ** 2)


def bound_loss(x, x_min, x_max, alpha=1):
    d = 0.5 * (x_max - x_min)
    return np.power(np.maximum(0, np.abs(x - (d + x_min)) - d), alpha)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x_min = 4
    x_max = 6

    x = np.linspace(0, 10, 100)
    y = bound_loss(x, x_min, x_max)
    plt.plot(x, y)
    plt.show()

    solution = pso(_fitness, num_weights=2)
    print(solution)
