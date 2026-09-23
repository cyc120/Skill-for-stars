"""Calculate reproducible apparent relationships between astronomical targets."""

import csv
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import astropy
from astropy import units as u
from astropy.config.paths import set_temp_cache
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    SkyCoord,
    get_body,
    solar_system_ephemeris,
)
from astropy.time import Time

from starskill.astropy_offline import offline_iers
from starskill.ephemeris_calculator import build_time_grid
from starskill.schemas import (
    AstronomicalRelationshipResult,
    AstronomicalRelationshipSample,
    AstronomicalRelationshipSettings,
    AstronomicalRelationshipTask,
    ResolvedAstronomicalTarget,
    SolarSystemRelationshipResult,
    SolarSystemRelationshipSample,
    SolarSystemRelationshipSettings,
    SolarSystemRelationshipTask,
)
from starskill.target_references import resolve_target_ref
from starskill.target_resolver import TargetBackend


RELATIONSHIP_CSV_COLUMNS = (
    "timestamp_local",
    "timestamp_utc",
    "moon_altitude_deg",
    "moon_azimuth_deg",
    "jupiter_altitude_deg",
    "jupiter_azimuth_deg",
    "angular_separation_deg",
)

ASTRONOMICAL_RELATIONSHIP_CSV_COLUMNS = (
    "timestamp_local",
    "timestamp_utc",
    "primary_altitude_deg",
    "primary_azimuth_deg",
    "primary_is_above_horizon",
    "secondary_altitude_deg",
    "secondary_azimuth_deg",
    "secondary_is_above_horizon",
    "angular_separation_deg",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_altaz(
    target: ResolvedAstronomicalTarget,
    *,
    times: Time,
    location: EarthLocation,
    frame: AltAz,
) -> SkyCoord:
    if target.motion == "dynamic":
        assert target.kind == "solar_system"
        return get_body(target.label.casefold(), times, location=location).transform_to(
            frame
        )

    assert target.ra_deg is not None and target.dec_deg is not None
    return SkyCoord(
        ra=target.ra_deg * u.deg,
        dec=target.dec_deg * u.deg,
        frame="icrs",
    ).transform_to(frame)


def calculate_astronomical_relationship(
    task: AstronomicalRelationshipTask,
    *,
    target_backend: TargetBackend | None = None,
    cache_dir: Path | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> AstronomicalRelationshipResult:
    """Calculate observer-specific geometric AltAz positions and separation."""
    primary = resolve_target_ref(
        task.primary,
        backend=target_backend,
        cache_dir=cache_dir,
        clock=clock,
    )
    secondary = resolve_target_ref(
        task.secondary,
        backend=target_backend,
        cache_dir=cache_dir,
        clock=clock,
    )
    points = build_time_grid(
        start=task.time_range.start,
        end=task.time_range.end,
        timezone_name=task.observer.timezone,
        interval_minutes=task.interval_minutes,
    )
    with TemporaryDirectory(prefix="starskill-astropy-") as astropy_cache_dir:
        with (
            set_temp_cache(astropy_cache_dir),
            offline_iers(),
            solar_system_ephemeris.set("builtin"),
        ):
            times = Time([point.utc for point in points], scale="utc")
            location = EarthLocation(
                lon=task.observer.longitude * u.deg,
                lat=task.observer.latitude * u.deg,
                height=0 * u.m,
            )
            frame = AltAz(
                obstime=times,
                location=location,
                pressure=0 * u.hPa,
            )
            primary_altaz = _to_altaz(
                primary,
                times=times,
                location=location,
                frame=frame,
            )
            secondary_altaz = _to_altaz(
                secondary,
                times=times,
                location=location,
                frame=frame,
            )
            separation = primary_altaz.separation(secondary_altaz)

    samples = [
        AstronomicalRelationshipSample(
            timestamp_local=point.local,
            timestamp_utc=point.utc,
            primary_altitude_deg=float(
                primary_altaz.alt[index].to_value(u.deg)
            ),
            primary_azimuth_deg=float(primary_altaz.az[index].to_value(u.deg)),
            primary_is_above_horizon=bool(primary_altaz.alt[index] >= 0 * u.deg),
            secondary_altitude_deg=float(
                secondary_altaz.alt[index].to_value(u.deg)
            ),
            secondary_azimuth_deg=float(secondary_altaz.az[index].to_value(u.deg)),
            secondary_is_above_horizon=bool(
                secondary_altaz.alt[index] >= 0 * u.deg
            ),
            angular_separation_deg=float(separation[index].to_value(u.deg)),
        )
        for index, point in enumerate(points)
    ]
    return AstronomicalRelationshipResult(
        task=task,
        primary=primary,
        secondary=secondary,
        settings=AstronomicalRelationshipSettings(
            calculated_at=clock(),
            astropy_version=astropy.__version__,
        ),
        samples=samples,
    )


def calculate_solar_system_relationship(
    task: SolarSystemRelationshipTask,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> SolarSystemRelationshipResult:
    """Adapt the generalized calculator to the legacy Moon/Jupiter contract."""
    calculated_at = clock()
    generic_task = AstronomicalRelationshipTask(
        primary={"kind": "solar_system", "body": "moon"},
        secondary={"kind": "solar_system", "body": "jupiter"},
        observer=task.observer,
        time_range=task.time_range,
        interval_minutes=task.interval_minutes,
    )
    generic_result = calculate_astronomical_relationship(
        generic_task,
        clock=lambda: calculated_at,
    )
    samples = [
        SolarSystemRelationshipSample(
            timestamp_local=sample.timestamp_local,
            timestamp_utc=sample.timestamp_utc,
            moon_altitude_deg=sample.primary_altitude_deg,
            moon_azimuth_deg=sample.primary_azimuth_deg,
            jupiter_altitude_deg=sample.secondary_altitude_deg,
            jupiter_azimuth_deg=sample.secondary_azimuth_deg,
            angular_separation_deg=sample.angular_separation_deg,
        )
        for sample in generic_result.samples
    ]
    return SolarSystemRelationshipResult(
        task=task,
        settings=SolarSystemRelationshipSettings(
            calculated_at=generic_result.settings.calculated_at,
            astropy_version=generic_result.settings.astropy_version,
        ),
        samples=samples,
    )


def write_astronomical_relationship_csv(
    result: AstronomicalRelationshipResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ASTRONOMICAL_RELATIONSHIP_CSV_COLUMNS,
        )
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(
                {
                    "timestamp_local": sample.timestamp_local.isoformat(),
                    "timestamp_utc": sample.timestamp_utc.isoformat(),
                    "primary_altitude_deg": f"{sample.primary_altitude_deg:.3f}",
                    "primary_azimuth_deg": f"{sample.primary_azimuth_deg:.3f}",
                    "primary_is_above_horizon": str(
                        sample.primary_is_above_horizon
                    ),
                    "secondary_altitude_deg": f"{sample.secondary_altitude_deg:.3f}",
                    "secondary_azimuth_deg": f"{sample.secondary_azimuth_deg:.3f}",
                    "secondary_is_above_horizon": str(
                        sample.secondary_is_above_horizon
                    ),
                    "angular_separation_deg": f"{sample.angular_separation_deg:.3f}",
                }
            )


def write_relationship_csv(
    result: SolarSystemRelationshipResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELATIONSHIP_CSV_COLUMNS)
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(
                {
                    "timestamp_local": sample.timestamp_local.isoformat(),
                    "timestamp_utc": sample.timestamp_utc.isoformat(),
                    "moon_altitude_deg": f"{sample.moon_altitude_deg:.3f}",
                    "moon_azimuth_deg": f"{sample.moon_azimuth_deg:.3f}",
                    "jupiter_altitude_deg": f"{sample.jupiter_altitude_deg:.3f}",
                    "jupiter_azimuth_deg": f"{sample.jupiter_azimuth_deg:.3f}",
                    "angular_separation_deg": f"{sample.angular_separation_deg:.3f}",
                }
            )


def write_relationship_json(
    result: SolarSystemRelationshipResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
