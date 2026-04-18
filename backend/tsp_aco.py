import numpy as np
import time
from typing import List, Tuple, Dict

# Distance matrix from the task
DISTANCE_MATRIX = np.array([
    [0,  10, 12, 11, 14],
    [10,  0, 13, 15,  8],
    [12, 13,  0,  9, 14],
    [11, 15,  9,  0, 16],
    [14,  8, 14, 16,  0]
])

# Initial pheromone matrix (all ones)
PHEROMONE_INIT = np.ones((5, 5))


class AntSystemTSP:
    """Standard Ant System (AS) for TSP"""

    def __init__(self, dist_matrix: np.ndarray, pheromone_init: np.ndarray,
                 n_ants: int = 10, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.5, Q: float = 100.0):
        self.dist = dist_matrix.copy().astype(float)
        self.n_cities = dist_matrix.shape[0]
        self.pheromone = pheromone_init.copy().astype(float)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        # Heuristic visibility (1/distance), avoid division by zero
        with np.errstate(divide='ignore'):
            self.eta = np.where(self.dist > 0, 1.0 / self.dist, 0.0)

    def _select_next_city(self, current: int, visited: List[int]) -> int:
        unvisited = [c for c in range(self.n_cities) if c not in visited]
        probs = []
        for c in unvisited:
            tau = self.pheromone[current][c] ** self.alpha
            eta = self.eta[current][c] ** self.beta
            probs.append(tau * eta)
        total = sum(probs)
        probs = [p / total for p in probs]
        return np.random.choice(unvisited, p=probs)

    def _construct_solution(self, start: int) -> List[int]:
        tour = [start]
        while len(tour) < self.n_cities:
            nxt = self._select_next_city(tour[-1], tour)
            tour.append(nxt)
        return tour

    def _tour_length(self, tour: List[int]) -> float:
        length = sum(self.dist[tour[i]][tour[(i + 1) % self.n_cities]]
                     for i in range(self.n_cities))
        return length

    def _update_pheromone(self, all_tours: List[List[int]], all_lengths: List[float]):
        # Evaporation
        self.pheromone *= (1 - self.rho)
        # Deposit
        for tour, length in zip(all_tours, all_lengths):
            deposit = self.Q / length
            for i in range(self.n_cities):
                a, b = tour[i], tour[(i + 1) % self.n_cities]
                self.pheromone[a][b] += deposit
                self.pheromone[b][a] += deposit

    def solve(self) -> Dict:
        start_time = time.time()
        best_tour = None
        best_length = float('inf')
        history = []

        for it in range(self.n_iterations):
            all_tours, all_lengths = [], []
            for ant in range(self.n_ants):
                start = np.random.randint(self.n_cities)
                tour = self._construct_solution(start)
                length = self._tour_length(tour)
                all_tours.append(tour)
                all_lengths.append(length)
                if length < best_length:
                    best_length = length
                    best_tour = tour[:]
            self._update_pheromone(all_tours, all_lengths)
            history.append({
                'iteration': it + 1,
                'best_length': best_length,
                'avg_length': float(np.mean(all_lengths)),
                'pheromone_max': float(np.max(self.pheromone)),
                'pheromone_min': float(np.min(self.pheromone))
            })

        elapsed = time.time() - start_time
        return {
            'algorithm': 'Ant System (AS)',
            'best_tour': [int(c) for c in best_tour],
            'best_tour_named': [f'City {c}' for c in best_tour],
            'best_length': float(best_length),
            'history': history,
            'time_seconds': round(elapsed, 4),
            'final_pheromone': self.pheromone.tolist()
        }


class MaxMinAntSystem(AntSystemTSP):
    """Max-Min Ant System (MMAS) for TSP"""

    def __init__(self, *args, tau_max: float = 6.0, tau_min: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.pheromone = np.full_like(self.pheromone, tau_max)
        self.algorithm_name = 'Max-Min Ant System (MMAS)'

    def _update_pheromone(self, all_tours: List[List[int]], all_lengths: List[float]):
        # Evaporation
        self.pheromone *= (1 - self.rho)

        # Only best ant deposits pheromone
        best_idx = int(np.argmin(all_lengths))
        best_tour = all_tours[best_idx]
        best_length = all_lengths[best_idx]
        deposit = self.Q / best_length
        for i in range(self.n_cities):
            a, b = best_tour[i], best_tour[(i + 1) % self.n_cities]
            self.pheromone[a][b] += deposit
            self.pheromone[b][a] += deposit

        # Clamp to [tau_min, tau_max]
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def solve(self) -> Dict:
        result = super().solve()
        result['algorithm'] = self.algorithm_name
        result['tau_max'] = self.tau_max
        result['tau_min'] = self.tau_min
        return result


class RankBasedAntSystem(AntSystemTSP):
    """Rank-Based Ant System (AS-rank) for TSP"""

    def __init__(self, *args, weight: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight  # Number of top ants that deposit pheromone
        self.algorithm_name = 'Rank-Based Ant System (AS-rank)'
        self.best_so_far_tour = None
        self.best_so_far_length = float('inf')

    def _update_pheromone(self, all_tours: List[List[int]], all_lengths: List[float]):
        # Evaporation
        self.pheromone *= (1 - self.rho)

        # Update best-so-far
        iter_best_idx = int(np.argmin(all_lengths))
        if all_lengths[iter_best_idx] < self.best_so_far_length:
            self.best_so_far_length = all_lengths[iter_best_idx]
            self.best_so_far_tour = all_tours[iter_best_idx][:]

        # Rank ants
        ranked_indices = np.argsort(all_lengths)
        
        # Top (weight - 1) ants deposit pheromone
        # Rank mu=1 is the best, mu=weight-1 is the last to deposit
        for rank in range(min(self.weight - 1, self.n_ants)):
            idx = ranked_indices[rank]
            tour = all_tours[idx]
            length = all_lengths[idx]
            # Contribution: (weight - rank - 1) * Q/L
            deposit = (self.weight - rank - 1) * (self.Q / length)
            for i in range(self.n_cities):
                a, b = tour[i], tour[(i + 1) % self.n_cities]
                self.pheromone[a][b] += deposit
                self.pheromone[b][a] += deposit

        # Best-so-far ant always deposits pheromone with highest weight
        bsf_deposit = self.weight * (self.Q / self.best_so_far_length)
        for i in range(self.n_cities):
            a, b = self.best_so_far_tour[i], self.best_so_far_tour[(i + 1) % self.n_cities]
            self.pheromone[a][b] += bsf_deposit
            self.pheromone[b][a] += bsf_deposit

    def solve(self) -> Dict:
        self.best_so_far_tour = None
        self.best_so_far_length = float('inf')
        result = super().solve()
        result['algorithm'] = self.algorithm_name
        result['weight'] = self.weight
        return result


def compare_algorithms() -> Dict:
    """Run both algorithms and return comparison"""
    np.random.seed(42)

    as_solver = AntSystemTSP(
        DISTANCE_MATRIX, PHEROMONE_INIT,
        n_ants=10, n_iterations=100,
        alpha=1.0, beta=2.0, rho=0.5, Q=100.0
    )
    as_result = as_solver.solve()

    np.random.seed(42)
    mmas_solver = MaxMinAntSystem(
        DISTANCE_MATRIX, PHEROMONE_INIT,
        n_ants=10, n_iterations=100,
        alpha=1.0, beta=2.0, rho=0.5, Q=100.0,
        tau_max=6.0, tau_min=0.1
    )
    mmas_result = mmas_solver.solve()

    np.random.seed(42)
    rank_solver = RankBasedAntSystem(
        DISTANCE_MATRIX, PHEROMONE_INIT,
        n_ants=10, n_iterations=100,
        alpha=1.0, beta=2.0, rho=0.5, Q=100.0,
        weight=6
    )
    rank_result = rank_solver.solve()

    results = {
        'as': as_result,
        'mmas': mmas_result,
        'rank': rank_result
    }
    
    # Simple winner comparison (best of three)
    all_res = [as_result, mmas_result, rank_result]
    best_res = min(all_res, key=lambda x: x['best_length'])
    
    comparison = {
        'as': as_result,
        'mmas': mmas_result,
        'rank': rank_result,
        'comparison': {
            'as_best_length': as_result['best_length'],
            'mmas_best_length': mmas_result['best_length'],
            'rank_best_length': rank_result['best_length'],
            'as_time': as_result['time_seconds'],
            'mmas_time': mmas_result['time_seconds'],
            'rank_time': rank_result['time_seconds'],
            'winner': best_res['algorithm']
        }
    }
    return comparison


if __name__ == '__main__':
    import json
    result = compare_algorithms()
    print(json.dumps(result['comparison'], indent=2))
    print(f"\nAS  Best Tour: {result['as']['best_tour']} | Length: {result['as']['best_length']}")
    print(f"MMAS Best Tour: {result['mmas']['best_tour']} | Length: {result['mmas']['best_length']}")