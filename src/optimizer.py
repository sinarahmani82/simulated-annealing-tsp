import math
import numpy as np
from typing import List, Tuple, Dict, Any
from src.tsp import TSPProblem

class SimulatedAnnealing:
    """Simulated Annealing optimizer for combinatorial optimization."""
    def __init__(self, problem: TSPProblem, initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995, min_temp: float = 1e-4):
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def optimize(self, seed: int = 42) -> Tuple[List[int], float, List[float]]:
        """
        Execute annealing loop using Metropolis-Hastings acceptance probability:
        P(accept) = exp(-delta_cost / temperature)
        """
        np.random.seed(seed)
        current_tour = self.problem.generate_random_tour(seed=seed)
        current_cost = self.problem.compute_tour_distance(current_tour)

        best_tour = current_tour.copy()
        best_cost = current_cost

        temperature = self.initial_temp
        cost_history = [best_cost]

        while temperature > self.min_temp:
            neighbor_tour = self.problem.get_2opt_neighbor(current_tour)
            neighbor_cost = self.problem.compute_tour_distance(neighbor_tour)
            delta_cost = neighbor_cost - current_cost

            # Metropolis acceptance criterion
            if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / temperature):
                current_tour = neighbor_tour
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour.copy()

            cost_history.append(best_cost)
            temperature *= self.cooling_rate

        return best_tour, best_cost, cost_history
