from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from starskill.light_pollution import BLACK_MARBLE_PROVIDER
from starskill.recommendations import HUMAN_REVIEW_ITEMS, recommend_tonight
from starskill.schemas import (
    AstronomicalTargetSource,
    EphemerisSettings,
    ExternalSource,
    LightPollutionResult,
    ObservationPlanResult,
    ObservationWindow,
    Observer,
    ResolvedAstronomicalTarget,
    ResolvedTarget,
    TargetSource,
    VisibilityCriteria,
    VisibilitySample,
    WeatherForecast,
    WeatherSample,
)


WINDOW_START = datetime(2026, 1, 10, 20, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 1, 10, 22, tzinfo=timezone.utc)


def make_plan() -> ObservationPlanResult:
    source = TargetSource(
        database="SIMBAD",
        service_url="https://example.test/simbad",
        accessed_at=WINDOW_START,
        from_cache=False,
    )
    catalog_target = ResolvedTarget(
        input_name="M42",
        query_name="M 42",
        canonical_name="M 42",
        ra_deg=83.8201,
        dec_deg=-5.3876,
        object_type="HII",
        aliases=["M 42"],
        source=source,
    )
    return ObservationPlanResult(
        target=ResolvedAstronomicalTarget(
            label="M 42",
            kind="simbad",
            motion="fixed_icrs",
            ra_deg=83.8201,
            dec_deg=-5.3876,
            source=AstronomicalTargetSource(
                provider="simbad",
                from_cache=False,
                accessed_at=WINDOW_START,
            ),
            catalog_target=catalog_target,
        ),
        observer=Observer(
            location_name="Beijing",
            longitude=116.4074,
            latitude=39.9042,
            timezone="Asia/Shanghai",
        ),
        interval_minutes=60,
        source_ephemeris_settings=EphemerisSettings(
            calculated_at=WINDOW_START,
            astropy_version="7.2.0",
        ),
        criteria=VisibilityCriteria(),
        samples=[
            VisibilitySample(
                timestamp_local=WINDOW_START,
                timestamp_utc=WINDOW_START,
                target_altitude_deg=40,
                target_azimuth_deg=180,
                sun_altitude_deg=-20,
                moon_altitude_deg=10,
                moon_separation_deg=90,
                moon_illumination_fraction=0.5,
                is_observable=True,
                rejection_reasons=[],
            )
        ],
        windows=[
            ObservationWindow(
                start_local=WINDOW_START,
                end_local=WINDOW_END,
                start_utc=WINDOW_START,
                end_utc=WINDOW_END,
                sample_count=3,
                peak_target_altitude_deg=40,
            )
        ],
    )


def make_weather(
    cloud_cover_percent: float | None,
    precipitation_mm: float | None,
    *,
    timestamp: datetime = WINDOW_START,
) -> WeatherForecast:
    return WeatherForecast(
        samples=[
            WeatherSample(
                timestamp_local=timestamp,
                cloud_cover_percent=cloud_cover_percent,
                precipitation_mm=precipitation_mm,
            )
        ],
        source=ExternalSource(
            provider="Open-Meteo",
            source_url="https://example.test/weather",
            accessed_at=WINDOW_START,
            from_cache=False,
            availability="fresh",
        ),
    )


def unavailable_weather() -> WeatherForecast:
    return WeatherForecast(
        samples=[],
        source=ExternalSource(
            provider="Open-Meteo",
            source_url="https://example.test/weather",
            accessed_at=WINDOW_START,
            from_cache=False,
            availability="unavailable",
            issue_code="external_data_network_error",
        ),
    )


def make_light(radiance: float | None = 18.5) -> LightPollutionResult:
    return LightPollutionResult(
        radiance=radiance,
        unit="nW cm-2 sr-1" if radiance is not None else None,
        source=ExternalSource(
            provider=BLACK_MARBLE_PROVIDER,
            source_url="https://example.test/light",
            accessed_at=WINDOW_START,
            from_cache=False,
            availability="fresh" if radiance is not None else "unavailable",
        ),
    )


def test_human_review_requires_input_location_and_time_confirmation() -> None:
    assert HUMAN_REVIEW_ITEMS[0] == (
        "Verify the target, date, timezone, longitude, and latitude."
    )


def test_recommendation_downgrades_heavy_cloud() -> None:
    result = recommend_tonight(make_plan(), make_weather(92, 0), make_light())

    assert result.recommendations[0].grade == "not_recommended"
    assert "云量预报 92%" in result.recommendations[0].reasons
    assert result.human_review == list(HUMAN_REVIEW_ITEMS)


def test_unavailable_weather_cannot_upgrade_a_window() -> None:
    result = recommend_tonight(make_plan(), unavailable_weather(), make_light(None))

    assert result.recommendations[0].grade == "caution"
    assert "天气预报不可用，候选窗口仅基于几何条件" in result.recommendations[0].reasons
    assert "光害静态指标不可用" in result.recommendations[0].reasons


def test_matching_weather_is_inclusive_and_static_radiance_never_upgrades_grade() -> None:
    result = recommend_tonight(
        make_plan(), make_weather(60, 0, timestamp=WINDOW_END), make_light(999),
    )

    recommendation = result.recommendations[0]
    assert recommendation.grade == "caution"
    assert "云量预报 60%" in recommendation.reasons
    assert "静态环境亮度指标：999 nW cm-2 sr-1" in recommendation.reasons


FALL_BACK_ZONE = ZoneInfo("America/New_York")
WINDOW_START_LOCAL = datetime(2026, 11, 1, 1, 30, tzinfo=FALL_BACK_ZONE)
WINDOW_END_LOCAL = datetime(2026, 11, 1, 4, 0, tzinfo=FALL_BACK_ZONE)
# The second pass of the repeated hour: 06:00Z, labelled 01:00-05:00.
REPEATED_HOUR_SAMPLE = datetime(2026, 11, 1, 1, 0, fold=1, tzinfo=FALL_BACK_ZONE)


def make_fall_back_plan() -> ObservationPlanResult:
    return make_plan().model_copy(
        update={
            "windows": [
                ObservationWindow(
                    start_local=WINDOW_START_LOCAL,
                    end_local=WINDOW_END_LOCAL,
                    start_utc=WINDOW_START_LOCAL.astimezone(timezone.utc),
                    end_utc=WINDOW_END_LOCAL.astimezone(timezone.utc),
                    sample_count=3,
                    peak_target_altitude_deg=40,
                )
            ]
        }
    )


def test_repeated_hour_weather_matches_by_instant_not_wall_clock() -> None:
    # The sample is in-window by instant but its wall-clock label reads earlier
    # than the window start, so a label comparison would drop it and degrade the
    # window to caution.
    assert REPEATED_HOUR_SAMPLE.astimezone(timezone.utc) > WINDOW_START_LOCAL.astimezone(
        timezone.utc
    )
    assert REPEATED_HOUR_SAMPLE.replace(tzinfo=None) < WINDOW_START_LOCAL.replace(tzinfo=None)

    result = recommend_tonight(
        make_fall_back_plan(),
        make_weather(20, 0, timestamp=REPEATED_HOUR_SAMPLE),
        make_light(),
    )

    assert "云量预报 20%" in result.recommendations[0].reasons


def test_missing_window_weather_is_caution_even_when_forecast_is_available() -> None:
    result = recommend_tonight(
        make_plan(),
        make_weather(0, 0, timestamp=datetime(2026, 1, 10, 23, tzinfo=timezone.utc)),
        make_light(),
    )

    assert result.recommendations[0].grade == "caution"
    assert "天气预报不可用，候选窗口仅基于几何条件" in result.recommendations[0].reasons
