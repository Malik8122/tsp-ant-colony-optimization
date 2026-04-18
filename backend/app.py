from flask import Flask, jsonify, request
from flask_cors import CORS
from tsp_aco import AntSystemTSP, MaxMinAntSystem, RankBasedAntSystem, compare_algorithms, DISTANCE_MATRIX, PHEROMONE_INIT
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({'status': 'TSP ACO API running', 'endpoints': ['/run_as', '/run_mmas', '/run_rank_as', '/compare']})


@app.route('/run_as', methods=['POST'])
def run_as():
    data = request.get_json(silent=True) or {}
    n_ants = data.get('n_ants', 10)
    n_iterations = data.get('n_iterations', 100)
    alpha = data.get('alpha', 1.0)
    beta = data.get('beta', 2.0)
    rho = data.get('rho', 0.5)
    Q = data.get('Q', 100.0)
    seed = data.get('seed', 42)
    np.random.seed(seed)

    solver = AntSystemTSP(DISTANCE_MATRIX, PHEROMONE_INIT,
                          n_ants=n_ants, n_iterations=n_iterations,
                          alpha=alpha, beta=beta, rho=rho, Q=Q)
    result = solver.solve()
    return jsonify(result)


@app.route('/run_mmas', methods=['POST'])
def run_mmas():
    data = request.get_json(silent=True) or {}
    n_ants = data.get('n_ants', 10)
    n_iterations = data.get('n_iterations', 100)
    alpha = data.get('alpha', 1.0)
    beta = data.get('beta', 2.0)
    rho = data.get('rho', 0.5)
    Q = data.get('Q', 100.0)
    tau_max = data.get('tau_max', 6.0)
    tau_min = data.get('tau_min', 0.1)
    seed = data.get('seed', 42)
    np.random.seed(seed)

    solver = MaxMinAntSystem(DISTANCE_MATRIX, PHEROMONE_INIT,
                             n_ants=n_ants, n_iterations=n_iterations,
                             alpha=alpha, beta=beta, rho=rho, Q=Q,
                             tau_max=tau_max, tau_min=tau_min)
    result = solver.solve()
    return jsonify(result)


@app.route('/run_rank_as', methods=['POST'])
def run_rank_as():
    data = request.get_json(silent=True) or {}
    n_ants = data.get('n_ants', 10)
    n_iterations = data.get('n_iterations', 100)
    alpha = data.get('alpha', 1.0)
    beta = data.get('beta', 2.0)
    rho = data.get('rho', 0.5)
    Q = data.get('Q', 100.0)
    weight = data.get('weight', 6)
    seed = data.get('seed', 42)
    np.random.seed(seed)

    solver = RankBasedAntSystem(DISTANCE_MATRIX, PHEROMONE_INIT,
                               n_ants=n_ants, n_iterations=n_iterations,
                               alpha=alpha, beta=beta, rho=rho, Q=Q,
                               weight=weight)
    result = solver.solve()
    return jsonify(result)


@app.route('/compare', methods=['GET'])
def compare():
    result = compare_algorithms()
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)