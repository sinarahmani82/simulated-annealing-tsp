import numpy as np
from typing import List, Tuple

class TSPProblem:
    """Represents a Traveling Salesperson Problem instance on 2D coordinates."""
    def __init__(self, coordinates: np.ndarray):
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Coordinates array must have shape (N, 2).")
        self.cities = coordinates
        self.num_cities = len(coordinates)
        self.dist_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        """Precompute pairwise Euclidean distance matrix for O(1) distance lookups."""
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def compute_tour_distance(self, tour: List[int]) -> float:
        """Calculate total closed-loop Euclidean tour length."""
        distance = sum(self.dist_matrix[tour[i], tour[(i + 1) % self.num_cities]] for i in range(self.num_cities))
        return float(distance)

    def generate_random_tour(self, seed: int = None) -> List[int]:
        """Generate a valid random permutation of city indices."""
        if seed is not None:
            np.random.seed(seed)
        return list(np.random.permutation(self.num_cities))

    def get_2opt_neighbor(self, tour: List[int]) -> List[int]:
        """Apply a 2-opt inversion move by reversing a segment of the tour."""
        new_tour = tour.copy()
        i, j = sorted(np.random.choice(self.num_cities, size=2, replace=False))
        new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
        return new_tour
