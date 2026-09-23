"""Calculate observer-specific target, Sun, and Moon ephemerides."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import astropy
from astropy import units as u
from astropy.config.paths import set_temp_cache
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time

from starskill.astropy_offline import offline_iers
from starskill.schemas import (
    AstronomicalTargetSource,
    EphemerisResult,
    EphemerisSample,
    EphemerisSettings,
    ObservationTask,
    ResolvedAstronomicalTarget,
    ResolvedTarget,
)


EPHEMERIS_CSV_COLUMNS = (
    "timestamp_local",
    "timestamp_utc",
    "target_altitude_deg",
    "target_azimuth_deg",
    "sun_altitude_deg",
    "moon_altitude_deg",
    "moon_separation_deg",
)


@dataclass(frozen=True)
class TimePoint:
    local: datetime
    utc: datetime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_time_grid(
    *,
    start: datetime,
    end: datetime,
    timezone_name: str,
    interval_minutes: int,
) -> list[TimePoint]:
    """Build an inclusive grid of uniformly spaced UTC instants with local labels.

    The cadence is held in UTC rather than local wall-clock time. Across a DST
    transition the local clock skips or repeats an hour, so advancing a local
    datetime by a fixed step emits nonexistent local times that resolve to the
    same instant as a real one: rows duplicated, UTC no longer monotonic, and
    the run still reports success. Uniform real time is what the AltAz samples
    need; the local label is derived from each instant for display only.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")

    local_zone = ZoneInfo(timezone_name)
    local_start = (
        start.replace(tzinfo=local_zone)
        if start.tzinfo is None
        else start.astimezone(local_zone)
    )
    local_end = (
        end.replace(tzinfo=local_zone)
        if end.tzinfo is None
        else end.astimezone(local_zone)
    )
    step = timedelta(minutes=interval_minutes)

    start_utc = local_start.astimezone(timezone.utc)
    end_utc = local_end.astimezone(timezone.utc)
    # Compare the instants, not the wall-clock fields: two aware datetimes sharing
    # one tzinfo object compare by their local fields, which is the order a DST gap
    # can invert. A wall-clock guard over a UTC loop would accept a range and then
    # return an empty grid for it.
    # Strictly later, because a gap can also collapse a window to a single instant:
    # local 02:30-03:30 on a spring-forward night is zero minutes of real time, and
    # a one-sample "window" would misrepresent it. The task schema agrees.
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")

    points: list[TimePoint] = []
    current = start_utc
    while current <= end_utc:
        points.append(TimePoint(local=current.astimezone(local_zone), utc=current))
        current += step
    return points


def calculate_ephemeris(
    task: ObservationTask,
    target: ResolvedAstronomicalTarget,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> EphemerisResult:
    """Calculate geometric AltAz samples with bundled, offline IERS data."""
    if isinstance(target, ResolvedTarget):
        target = _generalize_catalog_target(target)
    points = build_time_grid(
        start=task.time_range.start,
        end=task.time_range.end,
        timezone_name=task.observer.timezone,
        interval_minutes=task.interval_minutes,
    )
    with TemporaryDirectory(prefix="starskill-astropy-") as cache_dir:
        with set_temp_cache(cache_dir), offline_iers():
            times = Time([point.utc for point in points], scale="utc")
            location = EarthLocation(
                lon=task.observer.longitude * u.deg,
                lat=task.observer.latitude * u.deg,
                height=0 * u.m,
            )
            horizontal_frame = AltAz(
                obstime=times,
                location=location,
                pressure=0 * u.hPa,
            )
            target_altaz = _target_altaz(
                target,
                times,
                location,
                horizontal_frame,
            )
            sun_altaz = get_sun(times).transform_to(horizontal_frame)
            moon_altaz = get_body("moon", times, location=location).transform_to(
                horizontal_frame
            )
    moon_separation = target_altaz.separation(moon_altaz)

    samples = [
        EphemerisSample(
            timestamp_local=point.local,
            timestamp_utc=point.utc,
            target_altitude_deg=float(target_altaz.alt[index].to_value(u.deg)),
            target_azimuth_deg=float(target_altaz.az[index].to_value(u.deg)),
            sun_altitude_deg=float(sun_altaz.alt[index].to_value(u.deg)),
            moon_altitude_deg=float(moon_altaz.alt[index].to_value(u.deg)),
            moon_separation_deg=float(moon_separation[index].to_value(u.deg)),
        )
        for index, point in enumerate(points)
    ]
    return EphemerisResult(
        target=target,
        observer=task.observer,
        interval_minutes=task.interval_minutes,
        settings=EphemerisSettings(
            calculated_at=clock(),
            astropy_version=astropy.__version__,
        ),
        samples=samples,
    )


def _generalize_catalog_target(target: ResolvedTarget) -> ResolvedAstronomicalTarget:
    return ResolvedAstronomicalTarget(
        label=target.canonical_name,
        kind="simbad",
        motion="fixed_icrs",
        ra_deg=target.ra_deg,
        dec_deg=target.dec_deg,
        source=AstronomicalTargetSource(
            provider="simbad_cache" if target.source.from_cache else "simbad",
            from_cache=target.source.from_cache,
            accessed_at=target.source.accessed_at,
        ),
        catalog_target=target,
    )


def _target_altaz(
    target: ResolvedAstronomicalTarget,
    times: Time,
    location: EarthLocation,
    frame: AltAz,
) -> SkyCoord:
    if target.motion == "dynamic":
        assert target.kind == "solar_system"
        return get_body(
            target.label.casefold(), times, location=location
        ).transform_to(frame)

    assert target.ra_deg is not None and target.dec_deg is not None
    return SkyCoord(
        ra=target.ra_deg * u.deg,
        dec=target.dec_deg * u.deg,
        frame="icrs",
    ).transform_to(frame)


def write_ephemeris_csv(result: EphemerisResult, output_path: Path) -> None:
    """Write ephemeris samples with stable columns and degree precision."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPHEMERIS_CSV_COLUMNS)
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(
                {
                    "timestamp_local": sample.timestamp_local.isoformat(),
                    "timestamp_utc": sample.timestamp_utc.isoformat(),
                    "target_altitude_deg": f"{sample.target_altitude_deg:.6f}",
                    "target_azimuth_deg": f"{sample.target_azimuth_deg:.6f}",
                    "sun_altitude_deg": f"{sample.sun_altitude_deg:.6f}",
                    "moon_altitude_deg": f"{sample.moon_altitude_deg:.6f}",
                    "moon_separation_deg": f"{sample.moon_separation_deg:.6f}",
                }
            )


def write_ephemeris_json(result: EphemerisResult, output_path: Path) -> None:
    """Write the complete calculation result and its provenance as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
