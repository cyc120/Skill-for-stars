from copy import deepcopy
from datetime import datetime

import pytest
from pydantic import ValidationError

from starskill.schemas import (
    AstronomicalRelationshipTask,
    AstronomyImageSearchRequest,
    ObservingConditionsRequest,
    ObservationTask,
    SimbadTargetRef,
    SolarSystemRelationshipTask,
    TonightRecommendationRequest,
)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "task_type": "observation_plan",
        "target": "M42",
        "observer": {
            "location_name": "Beijing",
            "longitude": 116.4074,
            "latitude": 39.9042,
            "timezone": "Asia/Shanghai",
        },
        "time_range": {
            "start": "2026-01-10 18:00:00",
            "end": "2026-01-11 02:00:00",
        },
    }


def test_valid_observation_task_uses_documented_defaults(valid_payload: dict) -> None:
    task = ObservationTask.model_validate(valid_payload)

    assert task.target == SimbadTargetRef(kind="simbad", name="M42")
    assert task.interval_minutes == 10
    assert task.time_range.start == datetime(2026, 1, 10, 18, 0)
    assert task.output.language == "zh-CN"
    assert task.output.formats == ["json", "csv", "png", "md"]


def dst_emptied_payload(valid_payload: dict) -> dict:
    """A wall-clock-ordered window whose local hour a spring-forward skips."""
    payload = deepcopy(valid_payload)
    payload["observer"]["timezone"] = "America/New_York"
    payload["time_range"] = {
        "start": "2023-03-12 02:30:00",
        "end": "2023-03-12 03:00:00",
    }
    return payload


def test_observation_task_rejects_a_window_a_dst_gap_empties(valid_payload: dict) -> None:
    with pytest.raises(ValidationError, match="non-empty interval in real time"):
        ObservationTask.model_validate(dst_emptied_payload(valid_payload))


def test_observation_task_accepts_a_window_that_spans_a_dst_gap(valid_payload: dict) -> None:
    payload = deepcopy(valid_payload)
    payload["observer"]["timezone"] = "America/New_York"
    payload["time_range"] = {
        "start": "2023-03-12 00:00:00",
        "end": "2023-03-12 05:00:00",
    }

    task = ObservationTask.model_validate(payload)

    assert task.time_range.end > task.time_range.start


def test_relationship_tasks_reject_a_window_a_dst_gap_empties(valid_payload: dict) -> None:
    observer = dst_emptied_payload(valid_payload)["observer"]
    time_range = dst_emptied_payload(valid_payload)["time_range"]

    relationship_payloads = [
        (
            AstronomicalRelationshipTask,
            {
                "task_type": "astronomical_relationship",
                "primary": {"kind": "solar_system", "body": "mars"},
                "secondary": {"kind": "simbad", "name": "M31"},
                "observer": observer,
                "time_range": time_range,
            },
        ),
        (
            SolarSystemRelationshipTask,
            {
                "task_type": "solar_system_relationship",
                "targets": ["moon", "jupiter"],
                "observer": observer,
                "time_range": time_range,
            },
        ),
    ]

    for model, payload in relationship_payloads:
        with pytest.raises(ValidationError, match="non-empty interval in real time"):
            model.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("longitude", 181),
        ("longitude", -181),
        ("latitude", 91),
        ("latitude", -91),
    ],
)
def test_observer_rejects_coordinates_outside_earth_bounds(
    valid_payload: dict, field: str, value: float
) -> None:
    payload = deepcopy(valid_payload)
    payload["observer"][field] = value

    with pytest.raises(ValidationError):
        ObservationTask.model_validate(payload)


def test_observer_rejects_unknown_timezone(valid_payload: dict) -> None:
    payload = deepcopy(valid_payload)
    payload["observer"]["timezone"] = "Mars/Olympus_Mons"

    with pytest.raises(ValidationError, match="timezone"):
        ObservationTask.model_validate(payload)


def test_time_range_requires_end_after_start(valid_payload: dict) -> None:
    payload = deepcopy(valid_payload)
    payload["time_range"]["end"] = payload["time_range"]["start"]

    with pytest.raises(ValidationError, match="end"):
        ObservationTask.model_validate(payload)


def test_target_must_not_be_blank(valid_payload: dict) -> None:
    payload = deepcopy(valid_payload)
    payload["target"] = "   "

    with pytest.raises(ValidationError, match="target"):
        ObservationTask.model_validate(payload)


def test_task_type_only_accepts_observation_plan(valid_payload: dict) -> None:
    payload = deepcopy(valid_payload)
    payload["task_type"] = "horoscope"

    with pytest.raises(ValidationError, match="task_type"):
        ObservationTask.model_validate(payload)


@pytest.mark.parametrize("interval", [0, 121])
def test_interval_must_stay_within_supported_range(
    valid_payload: dict, interval: int
) -> None:
    payload = deepcopy(valid_payload)
    payload["interval_minutes"] = interval

    with pytest.raises(ValidationError, match="interval_minutes"):
        ObservationTask.model_validate(payload)


def test_unknown_input_fields_are_rejected(valid_payload: dict) -> None:
    payload = deepcopy(valid_payload)
    payload["targte"] = "M51"

    with pytest.raises(ValidationError, match="targte"):
        ObservationTask.model_validate(payload)


def test_tonight_request_has_conservative_defaults(valid_payload: dict) -> None:
    request = TonightRecommendationRequest.model_validate({"task": valid_payload})

    assert request.min_target_altitude_deg == 30.0
    assert request.max_sun_altitude_deg == -12.0


def test_conditions_request_rejects_an_empty_time_range(valid_payload: dict) -> None:
    with pytest.raises(ValidationError, match="end"):
        ObservingConditionsRequest.model_validate(
            {
                "observer": valid_payload["observer"],
                "time_range": {
                    "start": "2026-01-10T20:00:00+08:00",
                    "end": "2026-01-10T20:00:00+08:00",
                },
            }
        )


def test_generic_image_request_accepts_target_and_registered_provider() -> None:
    request = AstronomyImageSearchRequest.model_validate(
        {
            "target": {
                "kind": "coordinates",
                "label": "M31",
                "ra_deg": 10.684708,
                "dec_deg": 41.26875,
            },
            "field_of_view_arcmin": 12,
            "bands": ["g", "r", "i"],
            "provider_mode": "panstarrs",
        }
    )

    assert request.allowed_formats == ["jpeg", "png", "fits"]


@pytest.mark.parametrize(
    "observed_at",
    [None, "2026-01-10T10:00:00"],
)
def test_solar_system_image_request_requires_offset_aware_observed_at(
    observed_at: str | None,
) -> None:
    with pytest.raises(ValidationError, match="observed_at"):
        AstronomyImageSearchRequest.model_validate(
            {
                "target": {"kind": "solar_system", "body": "mars"},
                "observed_at": observed_at,
            }
        )


def test_solar_system_image_request_accepts_offset_aware_observed_at() -> None:
    request = AstronomyImageSearchRequest.model_validate(
        {
            "target": {"kind": "solar_system", "body": "mars"},
            "observed_at": "2026-01-10T10:00:00+08:00",
        }
    )

    assert request.observed_at is not None
    assert request.observed_at.utcoffset() is not None


def test_generic_image_request_rejects_unregistered_provider() -> None:
    with pytest.raises(ValidationError, match="provider_mode"):
        AstronomyImageSearchRequest.model_validate(
            {
                "target": {
                    "kind": "coordinates",
                    "label": "M31",
                    "ra_deg": 10.684708,
                    "dec_deg": 41.26875,
                },
                "provider_mode": "arbitrary_archive",
            }
        )
