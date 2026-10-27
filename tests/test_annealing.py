import pytest
import numpy as np
from src.tsp import TSPProblem
from src.optimizer import SimulatedAnnealing

def test_distance_matrix_and_tour_distance():
    """Verify Euclidean perimeter computation for a known unit square."""
    square_cities = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0]
    ])
    tsp = TSPProblem(square_cities)
    natural_tour = [0, 1, 2, 3]
    distance = tsp.compute_tour_distance(natural_tour)
    # A unit square closed loop perimeter is exactly 4.0
    assert pytest.approx(distance, rel=1e-5) == 4.0

def test_valid_permutation_after_2opt():
    """Ensure that 2-opt neighborhood perturbations maintain valid city permutations."""
    np.random.seed(42)
    coords = np.random.uniform(0, 50, size=(10, 2))
    tsp = TSPProblem(coords)
    initial_tour = tsp.generate_random_tour()

    neighbor_tour = tsp.get_2opt_neighbor(initial_tour)
    assert len(neighbor_tour) == 10
    assert sorted(neighbor_tour) == list(range(10))

def test_simulated_annealing_cost_reduction():
    """Verify that simulated annealing significantly reduces tour cost compared to random initialization."""
    np.random.seed(42)
    coords = np.random.uniform(0, 100, size=(20, 2))
    tsp = TSPProblem(coords)

    initial_tour = tsp.generate_random_tour(seed=42)
    initial_cost = tsp.compute_tour_distance(initial_tour)

    sa = SimulatedAnnealing(problem=tsp, initial_temp=500.0, cooling_rate=0.98, min_temp=1e-2)
    _, best_cost, history = sa.optimize(seed=42)

    assert best_cost < initial_cost
    assert len(history) > 100
