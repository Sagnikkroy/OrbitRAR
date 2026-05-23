"""
coverage.py
-----------
Given satellite ground tracks, compute:
  1. Which grid cells are covered at each timestep
  2. Max revisit time (gap between passes) for each grid cell

A ground point is "covered" by a satellite if the satellite's elevation
angle above that point's horizon is >= min_elevation_deg (typically 10°).
"""

import numpy as np
from propagator import propagate_constellation


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth (km)."""
    R = 6378.137
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_swath_radius(altitude_km, min_elevation_deg=10.0):
    """
    Compute the ground radius (km) visible from a satellite at given altitude.

    Uses spherical Earth geometry:
      - min_elevation_deg: minimum elevation angle for a ground station to
        consider itself 'covered' (10° is standard — avoids horizon effects)
    """
    Re = 6378.137
    elev_rad = np.radians(min_elevation_deg)
    # Earth central angle from sub-satellite point to edge of coverage
    rho = np.arccos(Re / (Re + altitude_km) * np.cos(elev_rad)) - elev_rad
    return Re * rho  # km


def compute_coverage_grid(tracks, grid_step_deg=2.0, min_elevation_deg=10.0):
    """
    For each (lat, lon) grid cell, determine at which timesteps it is covered
    by at least one satellite.

    Parameters
    ----------
    tracks          : output of propagate_constellation()
    grid_step_deg   : resolution of Earth grid (2° ≈ 220km, good balance)
    min_elevation_deg: minimum elevation angle to count as covered

    Returns
    -------
    grid_lats   : 1D array of grid latitudes
    grid_lons   : 1D array of grid longitudes
    revisit_times: 2D array (lat x lon) of max revisit time in minutes
    coverage_pct : 2D array (lat x lon) of % time covered
    """
    altitude_km = tracks[0]['altitude_km']
    swath_radius = compute_swath_radius(altitude_km, min_elevation_deg)
    n_steps = len(tracks[0]['times'])
    timestep_min = 1.0  # our timestep is 60 seconds = 1 minute

    # Build grid
    grid_lats = np.arange(-90, 90 + grid_step_deg, grid_step_deg)
    grid_lons = np.arange(-180, 180 + grid_step_deg, grid_step_deg)
    n_lat, n_lon = len(grid_lats), len(grid_lons)

    # coverage_mask[t, i, j] = True if grid point (i,j) covered at timestep t
    # To save memory, compute revisit on the fly
    revisit_times = np.zeros((n_lat, n_lon))
    coverage_count = np.zeros((n_lat, n_lon))

    # For each grid point, track last time it was covered
    last_covered = np.full((n_lat, n_lon), -1)  # -1 = never covered
    current_gap_start = np.zeros((n_lat, n_lon))
    max_gap = np.zeros((n_lat, n_lon))

    print(f"  Grid: {n_lat} x {n_lon} = {n_lat*n_lon} points")
    print(f"  Swath radius: {swath_radius:.1f} km")
    print(f"  Simulating {n_steps} timesteps...")

    for t in range(n_steps):
        if t % 120 == 0:
            print(f"    Timestep {t}/{n_steps}...")

        # Build coverage mask for this timestep
        covered_now = np.zeros((n_lat, n_lon), dtype=bool)

        for track in tracks:
            sat_lat = track['lats'][t]
            sat_lon = track['lons'][t]
            if np.isnan(sat_lat):
                continue

            # Vectorized distance check for all grid points
            # Use broadcasting: grid_lats is (n_lat,), sat_lat is scalar
            for j, glon in enumerate(grid_lons):
                dist = haversine_distance(grid_lats, glon, sat_lat, sat_lon)
                covered_now[:, j] |= (dist <= swath_radius)

        # Update coverage stats
        coverage_count += covered_now

        # Track gaps
        newly_covered = covered_now & (last_covered == -1) | covered_now
        gap = t - last_covered  # time since last covered (in timesteps)

        # Where covered NOW: update last_covered
        last_covered[covered_now] = t

        # Where NOT covered: accumulate gap
        not_covered = ~covered_now
        current_gap = np.where(last_covered >= 0, t - last_covered, 0)
        max_gap = np.maximum(max_gap, current_gap)

    revisit_times = max_gap * timestep_min  # convert to minutes
    coverage_pct = (coverage_count / n_steps) * 100

    return grid_lats, grid_lons, revisit_times, coverage_pct


def compute_coverage_score(tracks, region='india', grid_step_deg=3.0):
    """
    Single-number score for a constellation configuration.
    Lower = better (minimizing worst-case revisit time).

    Used by the optimizer as the objective function.
    """
    grid_lats, grid_lons, revisit_times, coverage_pct = compute_coverage_grid(
        tracks, grid_step_deg=grid_step_deg
    )

    if region == 'india':
        # India bounding box: lat 8-37°N, lon 68-98°E
        lat_mask = (grid_lats >= 8) & (grid_lats <= 37)
        lon_mask = (grid_lons >= 68) & (grid_lons <= 98)
        regional_revisit = revisit_times[np.ix_(lat_mask, lon_mask)]
    elif region == 'global':
        regional_revisit = revisit_times
    else:
        regional_revisit = revisit_times

    # Score = 95th percentile revisit time (robust to outliers)
    score = np.percentile(regional_revisit[regional_revisit > 0], 95)
    return score


if __name__ == "__main__":
    print("Testing coverage calculator...")
    print("Propagating 5-satellite constellation...")
    tracks = propagate_constellation(
        inclination_deg=55.0,
        raan_spacings_deg=[0, 72, 144, 216, 288],
        altitude_km=550.0,
        duration_hours=6.0,
        timestep_seconds=60.0,
    )

    grid_lats, grid_lons, revisit_times, coverage_pct = compute_coverage_grid(
        tracks, grid_step_deg=5.0  # coarse grid for quick test
    )

    print(f"\nResults:")
    print(f"  Global avg revisit time : {revisit_times[revisit_times>0].mean():.1f} min")
    print(f"  Global max revisit time : {revisit_times.max():.1f} min")
    print(f"  Global avg coverage     : {coverage_pct.mean():.1f}%")

    # India specifically
    lat_mask = (grid_lats >= 8) & (grid_lats <= 37)
    lon_mask = (grid_lons >= 68) & (grid_lons <= 98)
    india_revisit = revisit_times[np.ix_(lat_mask, lon_mask)]
    print(f"  India avg revisit time  : {india_revisit[india_revisit>0].mean():.1f} min")
    print(f"  India max revisit time  : {india_revisit.max():.1f} min")
    print("Coverage calculator OK ✓")