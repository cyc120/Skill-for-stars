"""Deterministic tonight recommendations from geometry and external evidence."""

from datetime import timezone

from starskill.schemas import (
    LightPollutionResult,
    ObservationPlanResult,
    RecommendationWindow,
    TonightRecommendationResult,
    WeatherForecast,
    WeatherSample,
)


HUMAN_REVIEW_ITEMS = (
    "Verify the target, date, timezone, longitude, and latitude.",
    "Check weather, cloud cover, transparency, and wind.",
    "Check buildings, trees, terrain, and local light pollution.",
    "Confirm telescope field of view, setup, and safety procedures.",
    "Compare at least one sky position with an independent tool.",
)


def recommend_tonight(
    geometry: ObservationPlanResult,
    weather: WeatherForecast,
    light_pollution: LightPollutionResult,
) -> TonightRecommendationResult:
    """Apply conservative weather rules without treating static radiance as a grade."""
    recommendations = [
        _recommend_window(
            window.start_local,
            window.end_local,
            window.start_utc,
            window.end_utc,
            weather.samples,
            weather,
            light_pollution,
        )
        for window in geometry.windows
    ]
    return TonightRecommendationResult(
        geometry=geometry,
        weather_forecast=weather,
        light_pollution=light_pollution,
        recommendations=recommendations,
        human_review=list(HUMAN_REVIEW_ITEMS),
        provenance=[weather.source, light_pollution.source],
    )


def _recommend_window(
    start,
    end,
    start_utc,
    end_utc,
    samples: list[WeatherSample],
    weather: WeatherForecast,
    light_pollution: LightPollutionResult,
) -> RecommendationWindow:
    # Match on instants, not wall-clock labels. Across a fall-back the local clock
    # repeats an hour, so an in-window sample's label can read earlier than the
    # window start; comparing labels would drop it. The labels stay on the result.
    matching = [
        sample
        for sample in samples
        if start_utc <= sample.timestamp_local.astimezone(timezone.utc) <= end_utc
    ]
    reasons: list[str]
    if (
        weather.source.availability == "unavailable"
        or not matching
        or any(
            sample.cloud_cover_percent is None or sample.precipitation_mm is None
            for sample in matching
        )
    ):
        grade = "caution"
        reasons = ["天气预报不可用，候选窗口仅基于几何条件"]
    else:
        grade, reasons = _grade_weather(matching)
    reasons.append(_light_reason(light_pollution))
    return RecommendationWindow(
        start_local=start,
        end_local=end,
        grade=grade,
        reasons=reasons,
    )


def _grade_weather(samples: list[WeatherSample]) -> tuple[str, list[str]]:
    clouds = [sample.cloud_cover_percent for sample in samples]
    precipitation = [sample.precipitation_mm for sample in samples]
    assert all(value is not None for value in clouds)
    assert all(value is not None for value in precipitation)
    highest_cloud = max(value for value in clouds if value is not None)
    wettest = max(value for value in precipitation if value is not None)
    reasons = [f"云量预报 {_format_number(highest_cloud)}%"]
    if wettest > 0:
        reasons.append(f"降水预报 {_format_number(wettest)} mm")
    if highest_cloud >= 85 or wettest > 0:
        return "not_recommended", reasons
    if highest_cloud >= 60:
        return "caution", reasons
    return "recommended", reasons


def _light_reason(light_pollution: LightPollutionResult) -> str:
    if light_pollution.radiance is None or not light_pollution.unit:
        return "光害静态指标不可用"
    return (
        "静态环境亮度指标："
        f"{_format_number(light_pollution.radiance)} {light_pollution.unit}"
    )


def _format_number(value: float) -> str:
    return f"{value:g}"
