"""
run_pipeline.py
---------------
Run the full Garud Constellation Optimizer pipeline:

  Step 1: Generate training data (simulate random configs)
  Step 2: Train the ML surrogate model
  Step 3: Run naive baseline
  Step 4: Run GA with full simulator (slow, small pop)
  Step 5: Run GA with surrogate model (fast, large pop)
  Step 6: Compare all three results
  Step 7: Save everything for the dashboard

Usage:
  python run_pipeline.py              # full pipeline
  python run_pipeline.py --quick      # quick test with small settings
"""

import numpy as np
import json
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from propagator import propagate_constellation
from coverage import compute_coverage_grid, compute_coverage_score
from optimizer import naive_baseline, genetic_algorithm, generate_training_data, objective, config_to_vector
from surrogate_model import SurrogateModel, surrogate_assisted_ga

os.makedirs('data', exist_ok=True)

def run_pipeline(quick=False):
    # ── Config ──────────────────────────────────────
    N_SATS        = 6
    ALT_KM        = 550.0
    N_SAMPLES     = 30  if quick else 150   # training data samples
    GA_POP        = 8   if quick else 20    # GA population size
    GA_GENS       = 5   if quick else 12    # GA generations
    SGA_POP       = 50  if quick else 200   # surrogate GA population
    SGA_GENS      = 10  if quick else 40    # surrogate GA generations
    DURATION_HRS  = 4.0 if quick else 8.0  # simulation duration
    GRID_STEP     = 5.0 if quick else 3.0  # grid resolution (degrees)

    print("=" * 65)
    print("  GARUD CONSTELLATION OPTIMIZER — Full Pipeline")
    print("=" * 65)
    print(f"  Satellites : {N_SATS}")
    print(f"  Altitude   : {ALT_KM} km")
    print(f"  Mode       : {'QUICK TEST' if quick else 'FULL RUN'}")
    print("=" * 65)

    results = {}

    # ── Step 1: Naive Baseline ───────────────────────
    print(f"\n{'─'*65}")
    print(f"STEP 1: Naive Baseline")
    print(f"{'─'*65}")
    t0 = time.time()
    inc_naive, raans_naive = naive_baseline(N_SATS, ALT_KM, inclination_deg=55.0)
    v_naive = config_to_vector(inc_naive, raans_naive)
    score_naive = objective(v_naive, N_SATS, ALT_KM, DURATION_HRS, GRID_STEP)
    print(f"  Inclination : {inc_naive}°")
    print(f"  RAANs       : {[round(r,1) for r in raans_naive]}")
    print(f"  Score       : {score_naive:.1f} min  ({time.time()-t0:.1f}s)")
    results['naive'] = {
        'score': score_naive, 'inclination': inc_naive, 'raans': raans_naive
    }

    # ── Step 2: Generate Training Data ──────────────
    print(f"\n{'─'*65}")
    print(f"STEP 2: Generating {N_SAMPLES} Training Samples for ML Surrogate")
    print(f"{'─'*65}")
    t0 = time.time()
    X, y = generate_training_data(
        n_sats=N_SATS, altitude_km=ALT_KM, n_samples=N_SAMPLES,
        duration_hours=DURATION_HRS, grid_step=GRID_STEP,
        save_path='data/training'
    )
    print(f"  Done in {time.time()-t0:.1f}s | "
          f"Score range: [{y.min():.1f}, {y.max():.1f}] min")

    # ── Step 3: Train Surrogate ──────────────────────
    print(f"\n{'─'*65}")
    print(f"STEP 3: Training ML Surrogate Model")
    print(f"{'─'*65}")
    surrogate = SurrogateModel()
    metrics = surrogate.train(X, y)
    surrogate.save('data/surrogate_model.pkl')
    results['surrogate_metrics'] = metrics

    # ── Step 4: GA with Full Simulator ──────────────
    print(f"\n{'─'*65}")
    print(f"STEP 4: Genetic Algorithm (Full Simulator)")
    print(f"{'─'*65}")
    t0 = time.time()
    ga_result = genetic_algorithm(
        n_sats=N_SATS, altitude_km=ALT_KM,
        population_size=GA_POP, n_generations=GA_GENS,
        duration_hours=DURATION_HRS, grid_step=GRID_STEP,
        verbose=True,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    results['ga_optimized'] = ga_result

    # ── Step 5: Surrogate-Assisted GA ───────────────
    print(f"\n{'─'*65}")
    print(f"STEP 5: Surrogate-Assisted Genetic Algorithm (ML-Accelerated)")
    print(f"{'─'*65}")
    t0 = time.time()
    sga_result = surrogate_assisted_ga(
        surrogate=surrogate,
        n_sats=N_SATS, altitude_km=ALT_KM,
        population_size=SGA_POP, n_generations=SGA_GENS,
        verbose=True,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    results['surrogate_ga'] = sga_result

    # ── Step 6: Generate Ground Tracks for Dashboard ─
    print(f"\n{'─'*65}")
    print(f"STEP 6: Computing Ground Tracks for Dashboard")
    print(f"{'─'*65}")

    best_result = sga_result if sga_result['verified_score_min'] < ga_result['best_score_min'] \
                  else ga_result

    for label, inc, raans in [
        ('naive', inc_naive, raans_naive),
        ('optimized', best_result['inclination'], best_result.get('raan_spacings', best_result.get('raan_spacings')))
    ]:
        tracks = propagate_constellation(
            inclination_deg=inc,
            raan_spacings_deg=raans,
            altitude_km=ALT_KM,
            duration_hours=DURATION_HRS,
            timestep_seconds=60.0,
        )
        grid_lats, grid_lons, revisit_times, coverage_pct = compute_coverage_grid(
            tracks, grid_step_deg=GRID_STEP
        )
        # Save track data (subsample for dashboard)
        track_data = []
        for t in tracks:
            step = max(1, len(t['lats']) // 200)  # max 200 points per sat
            track_data.append({
                'sat_id': t['sat_id'],
                'lats': t['lats'][::step].tolist(),
                'lons': t['lons'][::step].tolist(),
            })
        np.save(f'data/{label}_revisit.npy', revisit_times)
        np.save(f'data/{label}_coverage.npy', coverage_pct)
        with open(f'data/{label}_tracks.json', 'w') as f:
            json.dump(track_data, f)
        print(f"  {label}: avg revisit {revisit_times[revisit_times>0].mean():.1f} min, "
              f"max {revisit_times.max():.1f} min")

    np.save('data/grid_lats.npy', grid_lats)
    np.save('data/grid_lons.npy', grid_lons)

    # ── Step 7: Final Summary ────────────────────────
    print(f"\n{'='*65}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*65}")
    best_score = min(
        ga_result['best_score_min'],
        sga_result['verified_score_min']
    )
    improvement = (score_naive - best_score) / score_naive * 100
    print(f"  Naive baseline          : {score_naive:.1f} min worst-case revisit")
    print(f"  GA optimized            : {ga_result['best_score_min']:.1f} min")
    print(f"  Surrogate GA (verified) : {sga_result['verified_score_min']:.1f} min")
    print(f"  Best improvement        : {improvement:.1f}% over naive")
    print(f"\n  ML Surrogate R² score   : {metrics['r2_score']}")
    print(f"  ML Surrogate MAE        : {metrics['mae_minutes']} min")

    # Save all results for dashboard
    results['grid_step'] = GRID_STEP
    results['n_sats'] = N_SATS
    results['altitude_km'] = ALT_KM
    with open('data/pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  All data saved to ./data/")
    print(f"  Run dashboard: streamlit run dashboard/app.py")
    print(f"{'='*65}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (small settings, ~5 min)')
    args = parser.parse_args()
    run_pipeline(quick=args.quick)