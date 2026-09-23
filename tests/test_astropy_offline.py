"""Regression coverage for the offline astropy IERS policy."""

from __future__ import annotations

from datetime import datetime, timezone

from astropy.utils import iers
import pytest

import starskill
import starskill.schemas as schemas
from starskill.astropy_offline import offline_iers
from starskill.observation_planner import calculate_moon_illumination
from starskill.schemas import SkyChartRequest
from starskill.sky_chart import SkyChartService
from starskill.sky_chart_targets import SkyChartTargetResolver


# astropy 7.2 ships IERS-A predictions that stop in 2026, so any date in the
# spring of 2027 is beyond the packaged horizon and used to abort the
# UTC-to-UT1 step every calculation below depends on.
STALE_DATE = datetime(2027, 6, 1, 20, 0, tzinfo=timezone.utc)
STALE_DATE_LOCAL = "2027-06-01 20:00:00"
STALE_DATE_LOCAL_END = "2027-06-01 20:10:00"


def test_offline_iers_disables_download_and_stale_check() -> None:
    with offline_iers():
        assert iers.conf.auto_download is False
        assert iers.conf.auto_max_age is None


def test_offline_iers_restores_ambient_configuration() -> None:
    before = (iers.conf.auto_download, iers.conf.auto_max_age)

    with offline_iers():
        pass

    assert (iers.conf.auto_download, iers.conf.auto_max_age) == before


def test_offline_iers_survives_exceptions() -> None:
    before = (iers.conf.auto_download, iers.conf.auto_max_age)

    with pytest.raises(RuntimeError):
        with offline_iers():
            raise RuntimeError

    assert (iers.conf.auto_download, iers.conf.auto_max_age) == before


def test_moon_illumination_works_beyond_the_packaged_iers_table() -> None:
    values = calculate_moon_illumination([STALE_DATE])

    assert len(values) == 1
    assert 0.0 <= values[0] <= 1.0


def make_observation_task() -> schemas.ObservationTask:
    return schemas.ObservationTask.model_validate(
        {
            "task_type": "observation_plan",
            "target": {"kind": "solar_system", "body": "mars"},
            "observer": {
                "location_name": "北京",
                "longitude": 116.4074,
                "latitude": 39.9042,
                "timezone": "Asia/Shanghai",
            },
            "time_range": {"start": STALE_DATE_LOCAL, "end": STALE_DATE_LOCAL_END},
            "interval_minutes": 10,
        }
    )


def make_resolved_mars() -> schemas.ResolvedAstronomicalTarget:
    return schemas.ResolvedAstronomicalTarget.model_validate(
        {
            "label": "Mars",
            "kind": "solar_system",
            "motion": "dynamic",
            "source": {
                "provider": "astropy_builtin_ephemeris",
                "from_cache": False,
                "accessed_at": "2026-07-18T12:20:49Z",
            },
        }
    )


def test_ephemeris_works_beyond_the_packaged_iers_table() -> None:
    result = starskill.calculate_ephemeris(make_observation_task(), make_resolved_mars())

    assert len(result.samples) == 2
    assert -90.0 <= result.samples[0].target_altitude_deg <= 90.0


def test_relationship_works_beyond_the_packaged_iers_table() -> None:
    task = schemas.SolarSystemRelationshipTask.model_validate(
        {
            "task_type": "solar_system_relationship",
            "targets": ["moon", "jupiter"],
            "observer": {
                "location_name": "Shanghai",
                "longitude": 121.4737,
                "latitude": 31.2304,
                "timezone": "Asia/Shanghai",
            },
            "time_range": {"start": STALE_DATE_LOCAL, "end": STALE_DATE_LOCAL_END},
            "interval_minutes": 20,
        }
    )

    result = starskill.calculate_solar_system_relationship(task)

    assert result.samples
    assert 0.0 <= result.samples[0].angular_separation_deg <= 180.0


def test_sky_chart_renders_beyond_the_packaged_iers_table() -> None:
    class EmptyFullCache:
        def load_valid(self) -> None:
            return None

    service = SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_resolver=SkyChartTargetResolver(lambda _name: None),
        utc_clock=lambda: STALE_DATE,
    )
    request = SkyChartRequest.model_validate(
        {
            "observer": {
                "location_name": "Beijing",
                "longitude": 116.4074,
                "latitude": 39.9042,
                "timezone": "Asia/Shanghai",
            },
            "timestamp_local": "2027-06-01T20:00:00+08:00",
            "target": {"mode": "coordinates", "ra_deg": 83.822083, "dec_deg": -5.391111},
            "catalog_mode": "bundled",
        }
    )

    chart = service.render(request)

    assert chart.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert chart.metadata.calculation.iers_auto_download is False
