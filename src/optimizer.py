"""
optimizer.py
------------
Finds the best constellation configuration (inclination + RAAN spacings)
that minimizes worst-case revisit time over India.

Two approaches compared:
  1. Naive baseline   — equally spaced RAAN, fixed inclination
  2. Genetic Algorithm — evolves better configurations over generations

The ML surrogate model (neural network) speeds up the genetic algorithm
by predicting coverage scores without running the full physics simulation.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from propagator import propagate_constellation
from coverage import compute_coverage_score


# ─────────────────────────────────────────────
# 1. ENCODING: A constellation config as a vector
# ─────────────────────────────────────────────

def config_to_vector(inclination_deg, raan_spacings_deg):
    """Pack constellation config into a flat numpy array."""
    return np.array([inclination_deg] + list(raan_spacings_deg))


def vector_to_config(v):
    """Unpack flat array back to (inclination, [raan_spacings])."""
    return v[0], list(v[1:])


# ─────────────────────────────────────────────
# 2. OBJECTIVE FUNCTION
# ─────────────────────────────────────────────

def objective(v, n_sats, altitude_km, duration_hours=6.0, grid_step=4.0):
    """
    Evaluate a constellation config vector.
    Returns the 95th-percentile revisit time over India (lower = better).
    """
    inclination, raan_spacings = vector_to_config(v)

    # Clamp values to physical bounds
    inclination = np.clip(inclination, 20.0, 98.0)
    raan_spacings = [r % 360 for r in raan_spacings]

    tracks = propagate_constellation(
        inclination_deg=inclination,
        raan_spacings_deg=raan_spacings,
        altitude_km=altitude_km,
        duration_hours=duration_hours,
        timestep_seconds=60.0,
    )
    score = compute_coverage_score(tracks, region='india', grid_step_deg=grid_step)
    return score


# ─────────────────────────────────────────────
# 3. NAIVE BASELINE
# ─────────────────────────────────────────────

def naive_baseline(n_sats, altitude_km, inclination_deg=55.0):
    """
    Naive configuration: equally spaced RAAN, fixed inclination.
    This is what you'd do without any optimization.
    """
    raan_spacings = list(np.linspace(0, 360, n_sats, endpoint=False))
    return inclination_deg, raan_spacings


# ─────────────────────────────────────────────
# 4. GENETIC ALGORITHM
# ─────────────────────────────────────────────

def genetic_algorithm(
    n_sats,
    altitude_km,
    population_size=20,
    n_generations=15,
    mutation_rate=0.2,
    duration_hours=6.0,
    grid_step=4.0,
    verbose=True,
):
    """
    Genetic Algorithm to find optimal constellation configuration.

    How it works:
    - Start with a random population of constellation configs
    - Evaluate each config (compute coverage score)
    - Keep the best half (selection)
    - Create new configs by combining two parents (crossover)
    - Randomly tweak some values (mutation)
    - Repeat for n_generations

    Parameters
    ----------
    n_sats          : number of satellites in constellation
    altitude_km     : orbital altitude (fixed)
    population_size : number of configs evaluated per generation
    n_generations   : how many evolution cycles
    mutation_rate   : probability of randomly tweaking a parameter
    """
    dim = 1 + n_sats  # inclination + n RAAN values

    def random_individual():
        inc = np.random.uniform(30, 90)          # inclination between 30–90°
        raans = np.random.uniform(0, 360, n_sats) # random RAAN for each satellite
        return np.concatenate([[inc], raans])

    def crossover(parent1, parent2):
        """Single-point crossover: take first half from one parent, rest from other."""
        cut = np.random.randint(1, dim)
        child = np.concatenate([parent1[:cut], parent2[cut:]])
        return child

    def mutate(individual):
        """Randomly perturb some values."""
        ind = individual.copy()
        for i in range(dim):
            if np.random.rand() < mutation_rate:
                if i == 0:
                    ind[i] += np.random.uniform(-10, 10)  # inclination tweak
                else:
                    ind[i] += np.random.uniform(-30, 30)  # RAAN tweak
        return ind

    # Initialize population
    population = [random_individual() for _ in range(population_size)]
    best_score_history = []
    best_individual = None
    best_score = float('inf')

    if verbose:
        print(f"\n  GA started: {population_size} configs × {n_generations} generations")
        print(f"  Each evaluation = full 6-hour orbital simulation")

    for gen in range(n_generations):
        # Evaluate all individuals
        scores = []
        for ind in population:
            score = objective(ind, n_sats, altitude_km, duration_hours, grid_step)
            scores.append(score)

        scores = np.array(scores)

        # Track best
        gen_best_idx = np.argmin(scores)
        gen_best_score = scores[gen_best_idx]
        if gen_best_score < best_score:
            best_score = gen_best_score
            best_individual = population[gen_best_idx].copy()

        best_score_history.append(best_score)

        if verbose:
            print(f"  Gen {gen+1:2d}/{n_generations} | "
                  f"Best score: {best_score:.1f} min | "
                  f"Gen best: {gen_best_score:.1f} min | "
                  f"Pop avg: {scores.mean():.1f} min")

        # Selection: keep top 50%
        sorted_idx = np.argsort(scores)
        survivors = [population[i] for i in sorted_idx[:population_size//2]]

        # Create next generation
        next_population = survivors.copy()  # elitism: keep survivors
        while len(next_population) < population_size:
            p1, p2 = np.random.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[p1], survivors[p2])
            child = mutate(child)
            next_population.append(child)

        population = next_population

    inc_opt, raans_opt = vector_to_config(best_individual)
    inc_opt = np.clip(inc_opt, 20.0, 98.0)
    raans_opt = [r % 360 for r in raans_opt]

    return {
        'inclination': inc_opt,
        'raan_spacings': raans_opt,
        'best_score_min': best_score,
        'score_history': best_score_history,
    }


# ─────────────────────────────────────────────
# 5. GENERATE TRAINING DATA FOR ML SURROGATE
# ─────────────────────────────────────────────

def generate_training_data(n_sats, altitude_km, n_samples=200,
                            duration_hours=6.0, grid_step=4.0,
                            save_path=None):
    """
    Generate (config → coverage_score) pairs for training the surrogate model.

    Runs n_samples random constellation configs through the full physics
    simulation. This is the 'slow' step — the ML model learns to replicate
    these results instantly.
    """
    print(f"\nGenerating {n_samples} training samples...")
    print("(This is the slow step — we run the full simulator for each config)")

    X, y = [], []
    for i in range(n_samples):
        inc = np.random.uniform(30, 90)
        raans = np.random.uniform(0, 360, n_sats)
        v = np.concatenate([[inc], raans])
        score = objective(v, n_sats, altitude_km, duration_hours, grid_step)
        X.append(v)
        y.append(score)
        if (i+1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples} | score={score:.1f} min")

    X = np.array(X)
    y = np.array(y)

    if save_path:
        np.save(save_path + '_X.npy', X)
        np.save(save_path + '_y.npy', y)
        print(f"Saved to {save_path}_X.npy and {save_path}_y.npy")

    return X, y


if __name__ == "__main__":
    N_SATS = 6
    ALT_KM = 550.0

    print("=" * 60)
    print("GARUD CONSTELLATION OPTIMIZER")
    print("=" * 60)

    # Step 1: Naive baseline
    print("\n[1] Evaluating naive baseline (equally spaced RAAN)...")
    inc_naive, raans_naive = naive_baseline(N_SATS, ALT_KM)
    v_naive = config_to_vector(inc_naive, raans_naive)
    score_naive = objective(v_naive, N_SATS, ALT_KM)
    print(f"    Inclination : {inc_naive}°")
    print(f"    RAAN spacing: {[round(r,1) for r in raans_naive]}")
    print(f"    Score       : {score_naive:.1f} min (95th pct revisit time over India)")

    # Step 2: Genetic Algorithm
    print("\n[2] Running Genetic Algorithm optimizer...")
    result = genetic_algorithm(
        n_sats=N_SATS,
        altitude_km=ALT_KM,
        population_size=12,
        n_generations=8,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"  Naive baseline  : {score_naive:.1f} min worst-case revisit")
    print(f"  GA optimized    : {result['best_score_min']:.1f} min worst-case revisit")
    improvement = (score_naive - result['best_score_min']) / score_naive * 100
    print(f"  Improvement     : {improvement:.1f}%")
    print(f"\n  Optimal inclination : {result['inclination']:.2f}°")
    print(f"  Optimal RAANs       : {[round(r,1) for r in result['raan_spacings']]}")

    # Save results
    with open('../data/optimization_result.json', 'w') as f:
        json.dump({
            'naive_score': score_naive,
            'naive_inclination': inc_naive,
            'naive_raans': raans_naive,
            'optimized_score': result['best_score_min'],
            'optimized_inclination': result['inclination'],
            'optimized_raans': result['raan_spacings'],
            'score_history': result['score_history'],
            'n_sats': N_SATS,
            'altitude_km': ALT_KM,
        }, f, indent=2)
    print("\nResults saved to data/optimization_result.json")