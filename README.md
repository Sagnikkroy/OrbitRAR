# Garud Constellation Optimizer
### A Surrogate-Assisted ML Optimizer for LEO Satellite Constellation Design

Built as a project for Dhruva Space's **Project Garud** internship application.

---

## What This Project Does

Given N satellites to deploy in Low Earth Orbit, this tool finds the optimal
orbital configuration (inclination + RAAN spacing) that **minimizes coverage gaps
over India** — the core challenge for any constellation-scale satellite mission.

**The result:** 55%+ improvement in worst-case revisit time over a naive configuration.

---

## How It Works (Technical)

```
Random Configs ──► Full Physics Simulator ──► Coverage Scores
                         (SGP4 propagator)         (training data)
                                │
                                ▼
                    Train Neural Network (Surrogate)
                                │
                                ▼
         Genetic Algorithm ──► Surrogate ──► 1000x faster search
                                │
                                ▼
                    Best Config ──► Verify with real simulator
```

### Three Components

1. **Orbital Propagator** (`src/propagator.py`)
   - Uses SGP4 equations (same as NORAD) to propagate satellite orbits
   - Converts ECI coordinates → lat/lon ground tracks accounting for Earth's rotation
   - Handles N satellites simultaneously

2. **Coverage Calculator** (`src/coverage.py`)
   - Discretizes Earth into a lat/lon grid
   - For each grid cell: computes max gap between satellite passes (revisit time)
   - Uses spherical geometry to determine satellite visibility cone (~1665 km swath at 550km)
   - Objective: minimize 95th-percentile revisit time over India

3. **Optimizer** (`src/optimizer.py` + `src/surrogate_model.py`)
   - **Naive baseline**: equally spaced RAAN, fixed 55° inclination
   - **Genetic Algorithm**: evolves constellation configs using selection, crossover, mutation
   - **ML Surrogate**: 3-layer neural network trained on (config → coverage score) pairs
   - Surrogate makes GA ~1000x faster, enabling larger populations and more generations

---

## Project Structure

```
garud_optimizer/
├── src/
│   ├── propagator.py        # SGP4 orbit propagation + ground tracks
│   ├── coverage.py          # Coverage grid + revisit time computation
│   ├── optimizer.py         # Naive baseline + Genetic Algorithm
│   └── surrogate_model.py   # Neural network surrogate + surrogate GA
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
├── data/                    # Generated data (after running pipeline)
├── run_pipeline.py          # Main runner — runs everything
└── README.md
```

---

## Setup

```bash
# Clone and install dependencies
pip install hapsira sgp4 skyfield scipy numpy pandas scikit-learn \
            plotly streamlit requests

# Run quick test (~2 minutes)
python run_pipeline.py --quick

# Run full pipeline (~30-60 minutes, better results)
python run_pipeline.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Results

| Method | Worst-case Revisit (India) | Improvement |
|--------|--------------------------|-------------|
| Naive (equal RAAN spacing) | ~110 min | baseline |
| Genetic Algorithm | ~49 min | **55%** |
| Surrogate GA (full run) | ~40 min | **~64%** |

**ML Surrogate performance (full run):**
- R² score: ~0.85+
- MAE: ~8 minutes
- Speedup: ~1000x over full simulator

---

## Key Concepts

**SGP4**: Simplified General Perturbations model — the standard orbit propagator
used by NORAD and all space agencies. Accounts for Earth's oblateness, atmospheric drag.

**RAAN** (Right Ascension of Ascending Node): Controls the orientation of an orbit
around Earth. By spacing satellites' RAANs appropriately, you distribute coverage.

**Revisit Time**: How long a ground point waits between successive satellite passes.
For Earth observation, shorter = better. For Garud's constellation missions, this
determines how often India's ground stations can communicate with each satellite.

**Surrogate Model**: A fast approximation (neural network) of an expensive simulation.
Standard practice in aerospace engineering — used by NASA, ESA, Airbus for
computationally expensive simulations of aerodynamics, thermal, orbital mechanics.

---

## Relevance to Project Garud

Project Garud aims to manufacture and deploy 500+ kg-class satellites at scale
(up to 2/day). For any constellation mission using these satellites, the question
of *where* to place them in orbit directly determines mission value.

This tool provides:
- A principled, automated approach to constellation design
- Physics-accurate simulation using real orbital mechanics
- ML acceleration that makes large-scale optimization tractable
- A reusable framework applicable to Garud's telecom, EO, and security missions

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Orbit propagation | `sgp4` (same model as NORAD) |
| Earth/time calculations | `numpy`, `datetime` |
| ML surrogate | `scikit-learn` MLPRegressor |
| Optimization | Custom Genetic Algorithm |
| Dashboard | `streamlit`, `plotly` |

---

*Built with real orbital mechanics — no synthetic shortcuts.*