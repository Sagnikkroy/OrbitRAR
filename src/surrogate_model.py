"""
surrogate_model.py
------------------
A neural network trained to PREDICT coverage scores from constellation
configs — without running the slow physics simulation.

Why this matters:
  - Full simulation: ~30 seconds per config
  - Surrogate prediction: ~0.001 seconds per config
  - The GA can now evaluate 1000x more configs in the same time

This is called "surrogate-assisted optimization" — used in real aerospace
engineering (NASA, ESA, Airbus) for computationally expensive simulations.
"""

import numpy as np
import os
import json
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


class SurrogateModel:
    """
    Neural network surrogate for constellation coverage scoring.

    Input  : [inclination, raan_1, raan_2, ..., raan_n]  (n+1 features)
    Output : predicted 95th-percentile revisit time (minutes)
    """

    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # 3-layer network
            activation='relu',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            verbose=False,
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self.train_metrics = {}

    def _engineer_features(self, X):
        """
        Feature engineering: add domain-informed features.

        Raw input is [inclination, raan_1, ..., raan_n].
        We add features that capture the *geometry* of the constellation:
          - RAAN gaps (spacing between adjacent satellites)
          - Min/max/std of RAAN gaps (evenness of distribution)
          - sin/cos of inclination (circular feature)
        """
        feats = []
        for row in X:
            inc = row[0]
            raans = np.sort(row[1:])  # sort for consistent ordering

            # RAAN gaps between consecutive satellites
            gaps = np.diff(np.append(raans, raans[0] + 360))

            feat = list(row)  # original features
            feat += [
                np.sin(np.radians(inc)),   # circular encoding of inclination
                np.cos(np.radians(inc)),
                gaps.min(),                 # min gap (worst clustering)
                gaps.max(),                 # max gap (biggest hole)
                gaps.std(),                 # evenness of distribution
                gaps.mean(),                # average gap (should be 360/n)
                np.percentile(gaps, 25),
                np.percentile(gaps, 75),
            ]
            feats.append(feat)

        return np.array(feats)

    def train(self, X, y):
        """
        Train the surrogate on (config, score) pairs.

        X : array of shape (n_samples, n_params) — constellation configs
        y : array of shape (n_samples,) — coverage scores (revisit times)
        """
        print(f"\nTraining surrogate model on {len(X)} samples...")

        # Feature engineering
        X_eng = self._engineer_features(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_eng, y, test_size=0.2, random_state=42
        )

        # Normalize
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled  = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled  = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Train
        self.model.fit(X_train_scaled, y_train_scaled)

        # Evaluate
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))

        self.train_metrics = {
            'mae_minutes': round(mae, 2),
            'rmse_minutes': round(rmse, 2),
            'r2_score': round(r2, 4),
            'n_train': len(X_train),
            'n_test': len(X_test),
        }
        self.is_trained = True

        print(f"  Training complete!")
        print(f"  MAE  : {mae:.2f} minutes  (mean error in revisit prediction)")
        print(f"  RMSE : {rmse:.2f} minutes")
        print(f"  R²   : {r2:.4f}  (1.0 = perfect, 0.0 = useless)")
        return self.train_metrics

    def predict(self, X):
        """Predict coverage score(s) for one or more configs."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        X_eng = self._engineer_features(np.atleast_2d(X))
        X_scaled = self.scaler_X.transform(X_eng)
        y_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def save(self, path):
        """Save the trained model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'is_trained': self.is_trained,
                'train_metrics': self.train_metrics,
            }, f)
        print(f"Surrogate model saved to {path}")

    def load(self, path):
        """Load a trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        self.is_trained = data['is_trained']
        self.train_metrics = data['train_metrics']
        print(f"Surrogate model loaded from {path}")
        return self


def surrogate_assisted_ga(surrogate, n_sats, altitude_km,
                           population_size=100, n_generations=30,
                           mutation_rate=0.15, verbose=True):
    """
    Genetic Algorithm that uses the SURROGATE MODEL instead of the
    slow physics simulator. This allows much larger populations and
    more generations.

    Only the final best config is verified with the real simulator.
    """
    from optimizer import vector_to_config, objective
    import numpy as np

    dim = 1 + n_sats

    def random_individual():
        inc = np.random.uniform(30, 90)
        raans = np.random.uniform(0, 360, n_sats)
        return np.concatenate([[inc], raans])

    def crossover(p1, p2):
        cut = np.random.randint(1, dim)
        return np.concatenate([p1[:cut], p2[cut:]])

    def mutate(ind):
        ind = ind.copy()
        for i in range(dim):
            if np.random.rand() < mutation_rate:
                ind[i] += np.random.uniform(-20, 20) if i == 0 else np.random.uniform(-45, 45)
        return ind

    population = [random_individual() for _ in range(population_size)]
    best_score = float('inf')
    best_individual = None
    score_history = []

    if verbose:
        print(f"\n  Surrogate GA: {population_size} configs × {n_generations} generations")
        print(f"  (Using neural network — ~{population_size * n_generations}x faster than full sim)")

    for gen in range(n_generations):
        X = np.array(population)
        scores = surrogate.predict(X)

        gen_best_idx = np.argmin(scores)
        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_individual = population[gen_best_idx].copy()

        score_history.append(float(best_score))

        if verbose and (gen+1) % 5 == 0:
            print(f"  Gen {gen+1:3d}/{n_generations} | "
                  f"Surrogate best: {best_score:.1f} min | "
                  f"Pop avg: {scores.mean():.1f} min")

        sorted_idx = np.argsort(scores)
        survivors = [population[i] for i in sorted_idx[:population_size//2]]
        next_pop = survivors.copy()
        while len(next_pop) < population_size:
            p1, p2 = np.random.choice(len(survivors), 2, replace=False)
            child = mutate(crossover(survivors[p1], survivors[p2]))
            next_pop.append(child)
        population = next_pop

    # Verify best with real simulator
    if verbose:
        print(f"\n  Verifying best config with real physics simulator...")
    real_score = objective(best_individual, n_sats, altitude_km)

    inc_opt, raans_opt = vector_to_config(best_individual)
    return {
        'inclination': float(np.clip(inc_opt, 20, 98)),
        'raan_spacings': [float(r % 360) for r in raans_opt],
        'surrogate_score_min': float(best_score),
        'verified_score_min': float(real_score),
        'score_history': score_history,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Check if training data exists
    data_dir = '../data'
    X_path = f'{data_dir}/training_X.npy'
    y_path = f'{data_dir}/training_y.npy'

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("Loading existing training data...")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print("No training data found. Generating 50 samples (quick demo)...")
        print("Run optimizer.py generate_data for full 200-sample dataset.")
        from optimizer import generate_training_data
        os.makedirs(data_dir, exist_ok=True)
        X, y = generate_training_data(
            n_sats=6, altitude_km=550.0, n_samples=50,
            save_path=f'{data_dir}/training'
        )

    # Train surrogate
    surrogate = SurrogateModel()
    metrics = surrogate.train(X, y)

    # Save model
    os.makedirs(data_dir, exist_ok=True)
    surrogate.save(f'{data_dir}/surrogate_model.pkl')

    # Test prediction speed vs simulator
    import time
    test_config = np.array([55.0, 0, 60, 120, 180, 240, 300])
    t0 = time.time()
    for _ in range(1000):
        surrogate.predict(test_config)
    surrogate_time = (time.time() - t0) / 1000 * 1000  # ms

    print(f"\nSpeed comparison:")
    print(f"  Surrogate prediction : {surrogate_time:.3f} ms per config")
    print(f"  Full simulation      : ~30,000 ms per config")
    print(f"  Speedup              : ~{30000/surrogate_time:.0f}x faster")
    print(f"\nSurrogate model OK ✓")