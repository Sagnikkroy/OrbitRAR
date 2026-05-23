"""
propagator.py
-------------
Given orbital parameters for N satellites, propagate their orbits
and return ground tracks (lat/lon at each timestep).
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec


def _build_tle(sat_id, inclination_deg, raan_deg, altitude_km, mean_anomaly_deg=0.0):
    """
    Build a synthetic TLE string from simple orbital parameters.
    TLE format is fixed-width — every character position matters.
    """
    Re = 6378.137   # km
    mu = 398600.4418  # km^3/s^2
    a = Re + altitude_km
    n = np.sqrt(mu / a**3)            # rad/s
    n_revs_day = n * 86400 / (2*np.pi)  # revolutions per day

    # Epoch: 2024, day 152 (June 1)
    epoch = "24152.50000000"

    inc  = f"{inclination_deg:8.4f}"
    raan = f"{raan_deg:8.4f}"
    ecc  = "0001000"          # eccentricity ~0.001 (nearly circular)
    argp = "  0.0000"         # argument of perigee
    ma   = f"{mean_anomaly_deg:8.4f}"
    mm   = f"{n_revs_day:11.8f}"

    sid = str(sat_id).zfill(5)

    line1 = f"1 {sid}U 24001A   {epoch}  .00000000  00000-0  00000-0 0  9990"
    line2 = f"2 {sid} {inc} {raan} {ecc} {argp} {ma} {mm}    11"

    return line1, line2


def _jd_from_datetime(dt):
    """Convert UTC datetime to Julian Date."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days = (dt - j2000).total_seconds() / 86400.0
    return 2451545.0 + days


def _eci_to_latlon(r, dt):
    """Convert ECI position vector [x,y,z] km to lat/lon degrees."""
    x, y, z = r
    # Greenwich Sidereal Time
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days = (dt - j2000).total_seconds() / 86400.0
    gst = np.radians((280.46061837 + 360.98564736629 * days) % 360)
    # ECI → ECEF rotation
    xe =  x * np.cos(gst) + y * np.sin(gst)
    ye = -x * np.sin(gst) + y * np.cos(gst)
    ze = z
    r_mag = np.sqrt(xe**2 + ye**2 + ze**2)
    lat = np.degrees(np.arcsin(ze / r_mag))
    lon = np.degrees(np.arctan2(ye, xe))
    return lat, lon


def propagate_constellation(inclination_deg, raan_spacings_deg,
                             altitude_km, duration_hours=24.0,
                             timestep_seconds=60.0):
    """
    Propagate N satellites and return their ground tracks.

    Returns list of dicts with keys: sat_id, lats, lons, times
    """
    n_sats = len(raan_spacings_deg)
    epoch  = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    n_steps = int(duration_hours * 3600 / timestep_seconds)
    times = [epoch + timedelta(seconds=i * timestep_seconds) for i in range(n_steps)]

    results = []
    for i, raan in enumerate(raan_spacings_deg):
        # Space satellites along orbit too
        ma = (i * 360.0 / n_sats) % 360.0
        line1, line2 = _build_tle(i+1, inclination_deg, raan, altitude_km, ma)
        sat = Satrec.twoline2rv(line1, line2)

        lats, lons = [], []
        for t in times:
            jd = _jd_from_datetime(t)
            jd_whole = int(jd)
            jd_frac  = jd - jd_whole
            e, r, v = sat.sgp4(jd_whole, jd_frac)
            if e != 0 or any(np.isnan(r)):
                lats.append(np.nan); lons.append(np.nan)
            else:
                lat, lon = _eci_to_latlon(r, t)
                lats.append(lat); lons.append(lon)

        results.append({
            'sat_id': i+1,
            'lats': np.array(lats),
            'lons': np.array(lons),
            'times': times,
            'raan': raan,
            'inclination': inclination_deg,
            'altitude_km': altitude_km,
        })
    return results


if __name__ == "__main__":
    print("Testing propagator with 5 satellites @ 550km, 55° inclination...")
    tracks = propagate_constellation(
        inclination_deg=55.0,
        raan_spacings_deg=[0, 72, 144, 216, 288],
        altitude_km=550.0,
        duration_hours=2.0,
        timestep_seconds=60.0,
    )
    for t in tracks:
        valid = np.sum(~np.isnan(t['lats']))
        print(f"  Sat {t['sat_id']} | RAAN={t['raan']}° | {valid}/120 valid positions | "
              f"lat range [{t['lats'][~np.isnan(t['lats'])].min():.1f}, "
              f"{t['lats'][~np.isnan(t['lats'])].max():.1f}]°")
    print("Propagator OK ✓")