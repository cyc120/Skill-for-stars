"""Classify ephemeris samples and merge candidate observation windows."""

import csv
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy import units as u
from astropy.config.paths import set_temp_cache
from astropy.coordinates import get_body, get_sun
from astropy.time import Time

from starskill.astropy_offline import offline_iers
from starskill.schemas import (
    EphemerisResult,
    ObservationPlanResult,
    ObservationWindow,
    VisibilityCriteria,
    VisibilityRejectionReason,
    VisibilitySample,
)


VISIBILITY_CSV_COLUMNS = (
    "timestamp_local",
    "timestamp_utc",
    "target_altitude_deg",
    "target_azimuth_deg",
    "sun_altitude_deg",
    "moon_altitude_deg",
    "moon_separation_deg",
    "moon_illumination_fraction",
    "is_observable",
    "rejection_reasons",
)


def calculate_moon_illumination(timestamps: Sequence[datetime]) -> list[float]:
    """Calculate the geocentric illuminated fraction of the Moon."""
    if not timestamps:
        return []
    with TemporaryDirectory(prefix="starskill-astropy-") as cache_dir:
        with set_temp_cache(cache_dir), offline_iers():
            times = Time(list(timestamps), scale="utc")
            moon_vector = get_body("moon", times).cartesian.xyz.to(u.km)
            sun_vector = get_sun(times).cartesian.xyz.to(u.km)

    moon_to_sun = sun_vector - moon_vector
    moon_to_earth = -moon_vector
    cosine_phase = np.sum(moon_to_sun * moon_to_earth, axis=0) / (
        np.linalg.norm(moon_to_sun, axis=0)
        * np.linalg.norm(moon_to_earth, axis=0)
    )
    illuminated_fraction = (1.0 + np.clip(cosine_phase.to_value(u.one), -1, 1)) / 2
    return [float(value) for value in illuminated_fraction]


def _make_window(samples: list[VisibilitySample]) -> ObservationWindow:
    return ObservationWindow(
        start_local=samples[0].timestamp_local,
        end_local=samples[-1].timestamp_local,
        start_utc=samples[0].timestamp_utc,
        end_utc=samples[-1].timestamp_utc,
        sample_count=len(samples),
        peak_target_altitude_deg=max(sample.target_altitude_deg for sample in samples),
    )


def plan_observation(
    ephemeris: EphemerisResult,
    criteria: VisibilityCriteria | None = None,
) -> ObservationPlanResult:
    """Apply visibility thresholds and merge consecutive passing samples."""
    criteria = criteria or VisibilityCriteria()
    illumination = calculate_moon_illumination(
        [sample.timestamp_utc for sample in ephemeris.samples]
    )
    samples: list[VisibilitySample] = []
    windows: list[ObservationWindow] = []
    current_window: list[VisibilitySample] = []

    for ephemeris_sample, moon_illumination in zip(
        ephemeris.samples, illumination, strict=True
    ):
        reasons: list[VisibilityRejectionReason] = []
        if ephemeris_sample.target_altitude_deg < criteria.min_target_altitude_deg:
            reasons.append("target_below_minimum_altitude")
        if ephemeris_sample.sun_altitude_deg > criteria.max_sun_altitude_deg:
            reasons.append("sun_above_maximum_altitude")
        sample = VisibilitySample(
            **ephemeris_sample.model_dump(),
            moon_illumination_fraction=moon_illumination,
            is_observable=not reasons,
            rejection_reasons=reasons,
        )
        samples.append(sample)

        if sample.is_observable:
            current_window.append(sample)
        elif current_window:
            windows.append(_make_window(current_window))
            current_window = []

    if current_window:
        windows.append(_make_window(current_window))

    return ObservationPlanResult(
        target=ephemeris.target,
        observer=ephemeris.observer,
        interval_minutes=ephemeris.interval_minutes,
        source_ephemeris_settings=ephemeris.settings,
        criteria=criteria,
        samples=samples,
        windows=windows,
    )


def write_visibility_csv(plan: ObservationPlanResult, output_path: Path) -> None:
    """Write visibility evidence and rule outcomes using a stable CSV contract."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VISIBILITY_CSV_COLUMNS)
        writer.writeheader()
        for sample in plan.samples:
            writer.writerow(
                {
                    "timestamp_local": sample.timestamp_local.isoformat(),
                    "timestamp_utc": sample.timestamp_utc.isoformat(),
                    "target_altitude_deg": f"{sample.target_altitude_deg:.6f}",
                    "target_azimuth_deg": f"{sample.target_azimuth_deg:.6f}",
                    "sun_altitude_deg": f"{sample.sun_altitude_deg:.6f}",
                    "moon_altitude_deg": f"{sample.moon_altitude_deg:.6f}",
                    "moon_separation_deg": f"{sample.moon_separation_deg:.6f}",
                    "moon_illumination_fraction": (
                        f"{sample.moon_illumination_fraction:.6f}"
                    ),
                    "is_observable": str(sample.is_observable).lower(),
                    "rejection_reasons": ";".join(sample.rejection_reasons),
                }
            )


def write_observation_plan_json(
    plan: ObservationPlanResult,
    output_path: Path,
) -> None:
    """Write the complete plan, its source settings, and rule outcomes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
