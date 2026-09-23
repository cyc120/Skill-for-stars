from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from threading import Event
from types import SimpleNamespace
from typing import Iterable, Mapping

from fastapi.testclient import TestClient
import pytest

import starskill.cli as cli_module
from starskill.schemas import ResolvedTarget, SkyChartRequest, TargetSource
from starskill.sky_chart import RenderedSkyChart, SkyChartService
from starskill.sky_chart_catalog import (
    FullCatalogUnavailableError,
    HygSource,
)
import starskill.sky_chart_catalog as sky_chart_catalog_module
import starskill.target_resolver as target_resolver_module
from starskill.target_resolver import target_cache_path
import starskill.web_api as web_api_module
from starskill.web_api import FixedWindowRateLimiter, create_web_app
from tests.test_mcp_server import (
    load_observation_payload,
    make_service_with_fake_outreach_providers,
)


def valid_sky_chart_payload(*, catalog_mode: str = "bundled") -> dict[str, object]:
    return {
        "observer": {
            "location_name": "Beijing",
            "longitude": 116.4074,
            "latitude": 39.9042,
            "timezone": "Asia/Shanghai",
        },
        "timestamp_local": "2026-01-10T20:00:00+08:00",
        "target": {"mode": "name", "name": "M42"},
        "catalog_mode": catalog_mode,
    }


@pytest.fixture(scope="module")
def rendered_chart() -> RenderedSkyChart:
    return SkyChartService(
        utc_clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    ).render(SkyChartRequest.model_validate(valid_sky_chart_payload()))


class FakeSkyChartService:
    def __init__(self, chart: RenderedSkyChart) -> None:
        self.chart = chart

    def render(self, request: SkyChartRequest) -> RenderedSkyChart:
        return self.chart


class FakeFullCacheSkyChartService:
    def render(self, request: SkyChartRequest) -> RenderedSkyChart:
        raise FullCatalogUnavailableError("cache missing at /private/catalog")


class BrokenSkyChartService:
    def render(self, request: SkyChartRequest) -> RenderedSkyChart:
        raise RuntimeError("cache directory /private/config should not be exposed")


class BrokenService:
    def get_observing_conditions(self, request: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("cache directory /private/config should not be exposed")


class BytesCatalogFetcher:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def stream(
        self, url: str, *, max_bytes: int
    ) -> tuple[int, Mapping[str, str], Iterable[bytes]]:
        assert url.startswith("https://")
        assert len(self.body) <= max_bytes
        return 200, {"Content-Length": str(len(self.body))}, (self.body,)


def make_client(
    tmp_path: Path,
    rendered_chart: RenderedSkyChart,
    **kwargs: object,
) -> TestClient:
    return TestClient(
        create_web_app(
            make_service_with_fake_outreach_providers(tmp_path),
            FakeSkyChartService(rendered_chart),
            **kwargs,
        )
    )


def test_root_serves_packaged_same_origin_page(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "type=\"range\"" in response.text
    assert "/v1/sky-chart/render" in response.text
    assert "__DEFAULT_TIMESTAMP_LOCAL__" not in response.text
    assert all(
        token not in response.text.lower()
        for token in ("npm", "docker", "stellarium", "cdn", "http://", "https://")
    )


def test_page_contains_complete_controls_and_bounded_browser_lifecycle(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    page = make_client(tmp_path, rendered_chart).get("/").text

    for token in (
        'id="location-name"',
        'id="longitude"',
        'id="latitude"',
        'id="timezone"',
        'type="datetime-local"',
        'value="name"',
        'value="coordinates"',
        'id="target-name"',
        'id="target-ra"',
        'id="target-dec"',
        '<option value="auto"',
        '<option value="bundled"',
        '<option value="full"',
        'min="-720"',
        'max="720"',
        'step="15"',
        'aria-live="polite"',
        'id="control-rail"',
        'id="rail-toggle"',
        'aria-controls="control-rail"',
        'aria-expanded="true"',
        "AbortController",
        "250",
        "baseTimestamp",
        "longOffset",
    ):
        assert token in page
    assert re.search(r'<a[^>]+id="export-png"[^>]+aria-disabled="true"', page)
    assert re.search(r'<a[^>]+id="export-json"[^>]+aria-disabled="true"', page)
    assert "innerHTML" not in page


def test_docs_openapi_and_cors_are_disabled(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart)

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    response = client.get("/healthz", headers={"Origin": "https://foreign.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_render_and_both_exports_share_one_opaque_id(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart)

    response = client.post("/v1/sky-chart/render", json=valid_sky_chart_payload())

    assert response.status_code == 200
    body = response.json()
    assert re.fullmatch(r"[A-Za-z0-9_-]{32}", body["render_id"])
    assert body["png_url"] == f"/v1/sky-chart/renders/{body['render_id']}.png"
    assert body["json_url"] == f"/v1/sky-chart/renders/{body['render_id']}.json"
    png = client.get(body["png_url"])
    assert png.headers["content-type"] == "image/png"
    assert png.headers["content-disposition"] == (
        f'attachment; filename="starskill-sky-chart-{body["render_id"]}.png"'
    )
    metadata_response = client.get(body["json_url"])
    assert metadata_response.headers["content-type"] == "application/json"
    assert metadata_response.headers["content-disposition"] == (
        f'attachment; filename="starskill-sky-chart-{body["render_id"]}.json"'
    )
    assert metadata_response.json()["render_id"] == body["render_id"]


def test_sky_chart_declared_and_actual_body_limits_are_16_kib(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart)
    too_large = b"x" * (16 * 1024 + 1)

    declared = client.post(
        "/v1/sky-chart/render",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(len(too_large))},
    )
    actual = client.post(
        "/v1/sky-chart/render",
        content=too_large,
        headers={"content-type": "application/json", "content-length": "1"},
    )

    assert declared.status_code == 413
    assert declared.json() == {"detail": "Request body too large"}
    assert actual.status_code == 413
    assert actual.json() == {"detail": "Request body too large"}


def test_legacy_body_limit_remains_one_mib(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart)

    response = client.post(
        "/v1/conditions",
        content=b'{"observer":"' + b"x" * (1024 * 1024) + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_sky_chart_busy_and_invalid_ids_are_stable(
    tmp_path: Path, rendered_chart: RenderedSkyChart, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def timeout_wait(awaitable: object, timeout: float) -> None:
        assert timeout == 10
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError

    monkeypatch.setattr(web_api_module.asyncio, "wait_for", timeout_wait)
    client = make_client(tmp_path, rendered_chart)

    malformed = client.get("/v1/sky-chart/renders/not-valid!.png")
    missing = client.get("/v1/sky-chart/renders/missing.json")
    assert malformed.status_code == 404
    assert malformed.json() == {
        "detail": "Render not found"
    }
    assert missing.status_code == 404
    assert missing.json() == {
        "detail": "Render not found"
    }
    busy = client.post("/v1/sky-chart/render", json=valid_sky_chart_payload())
    assert busy.status_code == 503
    assert busy.json() == {"detail": "Renderer busy; retry shortly"}


def test_render_limiter_allows_30th_and_rejects_31st_without_changing_legacy_budget(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart, requests_per_minute=100)

    statuses = [
        client.post("/v1/sky-chart/render", json=valid_sky_chart_payload()).status_code
        for _index in range(31)
    ]

    assert statuses[:30] == [200] * 30
    assert statuses[30] == 429
    assert client.get("/healthz").status_code == 200


def test_malformed_render_consumes_only_the_dedicated_render_budget(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    legacy_limiter = FixedWindowRateLimiter(requests_per_minute=1)
    render_limiter = FixedWindowRateLimiter(requests_per_minute=1)
    client = make_client(
        tmp_path,
        rendered_chart,
        rate_limiter=legacy_limiter,
        sky_chart_rate_limiter=render_limiter,
    )

    malformed = client.post("/v1/sky-chart/render", json={"invalid": True})
    limited = client.post(
        "/v1/sky-chart/render", json=valid_sky_chart_payload()
    )

    assert malformed.status_code == 422
    assert malformed.json() == {"detail": "Invalid sky-chart request"}
    assert limited.status_code == 429
    assert limited.json() == {"detail": "Too many requests"}
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").status_code == 429


def test_chunked_render_body_stops_reading_as_soon_as_16_kib_is_exceeded(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    sky_service = FakeSkyChartService(rendered_chart)
    render_calls: list[SkyChartRequest] = []
    original_render = sky_service.render

    def record_render(request: SkyChartRequest) -> RenderedSkyChart:
        render_calls.append(request)
        return original_render(request)

    sky_service.render = record_render  # type: ignore[method-assign]
    app = create_web_app(
        make_service_with_fake_outreach_providers(tmp_path),
        sky_service,
    )
    request_events = [
        {"type": "http.request", "body": b"x" * 9_000, "more_body": True},
        {"type": "http.request", "body": b"y" * 9_000, "more_body": True},
        {"type": "http.request", "body": b"must-not-be-read", "more_body": False},
    ]
    receive_calls = 0
    response_events: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal receive_calls
        receive_calls += 1
        if request_events:
            return request_events.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        response_events.append(message)

    async def invoke() -> None:
        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/sky-chart/render",
                "raw_path": b"/v1/sky-chart/render",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        )

    asyncio.run(invoke())

    start = next(event for event in response_events if event["type"] == "http.response.start")
    body = b"".join(
        event.get("body", b"")  # type: ignore[arg-type]
        for event in response_events
        if event["type"] == "http.response.body"
    )
    assert start["status"] == 413
    assert json.loads(body) == {"detail": "Request body too large"}
    assert receive_calls == 2
    assert render_calls == []


def test_render_does_not_consume_the_legacy_request_budget(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(
        tmp_path,
        rendered_chart,
        requests_per_minute=1,
        sky_chart_requests_per_minute=30,
    )

    assert client.post(
        "/v1/sky-chart/render", json=valid_sky_chart_payload()
    ).status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").status_code == 429


def test_render_expiry_uses_injected_monotonic_clock(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    now = [0.0]
    client = make_client(tmp_path, rendered_chart, monotonic_clock=lambda: now[0])
    body = client.post("/v1/sky-chart/render", json=valid_sky_chart_payload()).json()

    now[0] = 899.0
    assert client.get(body["png_url"]).status_code == 200
    now[0] = 900.0
    expired = client.get(body["json_url"])
    assert expired.status_code == 404
    assert expired.json() == {"detail": "Render not found"}


def test_shutdown_clears_app_owned_render_store(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    app = create_web_app(
        make_service_with_fake_outreach_providers(tmp_path),
        FakeSkyChartService(rendered_chart),
    )
    with TestClient(app) as client:
        body = client.post("/v1/sky-chart/render", json=valid_sky_chart_payload()).json()
        assert app.state.render_store.get(body["render_id"]) is not None

    assert app.state.render_store.get(body["render_id"]) is None


def test_invalid_and_full_cache_requests_return_stable_422(tmp_path: Path) -> None:
    invalid_client = TestClient(
        create_web_app(
            make_service_with_fake_outreach_providers(tmp_path),
            FakeFullCacheSkyChartService(),
        )
    )

    invalid_model = invalid_client.post(
        "/v1/sky-chart/render",
        json={**valid_sky_chart_payload(), "timestamp_local": "2026-01-10T20:00:00"},
    )
    full_cache = invalid_client.post(
        "/v1/sky-chart/render", json=valid_sky_chart_payload(catalog_mode="full")
    )

    assert invalid_model.status_code == 422
    assert invalid_model.json() == {"detail": "Invalid sky-chart request"}
    assert full_cache.status_code == 422
    assert full_cache.json() == {"detail": "Invalid sky-chart request"}
    assert "private" not in full_cache.text


def test_render_exceptions_do_not_leak_paths(tmp_path: Path) -> None:
    client = TestClient(
        create_web_app(
            make_service_with_fake_outreach_providers(tmp_path), BrokenSkyChartService()
        )
    )

    response = client.post("/v1/sky-chart/render", json=valid_sky_chart_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "private" not in response.text


def test_existing_v1_response_shapes_and_stellarium_redaction(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    client = make_client(tmp_path, rendered_chart)
    task = load_observation_payload()

    conditions = client.post(
        "/v1/conditions",
        json={"observer": task["observer"], "time_range": task["time_range"]},
    )
    recommendation = client.post("/v1/recommendations/tonight", json={"task": task})
    nasa = client.get("/v1/nasa/apod?date=2026-01-01")
    stellarium = client.post(
        "/v1/stellarium/sync",
        json={
            "observer": task["observer"],
            "timestamp": "2026-07-23T20:00:00+08:00",
            "target": "M 42",
        },
    )

    assert conditions.status_code == 200
    assert "samples" in conditions.json()
    assert "source" in conditions.json()
    assert recommendation.status_code == 200
    assert recommendation.json()["human_review"]
    assert "run_id" not in recommendation.json()
    assert "resources" not in recommendation.json()
    assert nasa.status_code == 200
    assert "source" in nasa.json()
    assert stellarium.status_code == 200
    assert "base_url" not in stellarium.json()


def test_legacy_validation_rate_limit_and_generic_errors_remain_stable(
    tmp_path: Path, rendered_chart: RenderedSkyChart
) -> None:
    now = [100.0]
    limiter = FixedWindowRateLimiter(
        requests_per_minute=1, monotonic_clock=lambda: now[0]
    )
    limited_client = make_client(tmp_path, rendered_chart, rate_limiter=limiter)
    assert limited_client.get("/healthz").status_code == 200
    limited = limited_client.get("/healthz")
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "20"
    now[0] = 120.0
    assert limited_client.get("/healthz").status_code == 200

    service = make_service_with_fake_outreach_providers(tmp_path)
    validation_client = TestClient(create_web_app(service, FakeSkyChartService(rendered_chart)))
    assert validation_client.post("/v1/conditions", json={"observer": {}}).status_code == 422
    assert validation_client.get("/v1/nasa/apod?date=not-a-date").status_code == 422

    broken_client = TestClient(create_web_app(BrokenService(), FakeSkyChartService(rendered_chart)))
    response = broken_client.post("/v1/conditions", json={})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "private" not in response.text


def test_rate_limiter_prunes_expired_windows() -> None:
    now = [0.0]
    limiter = FixedWindowRateLimiter(
        requests_per_minute=1, monotonic_clock=lambda: now[0]
    )

    for window in range(100):
        now[0] = float(window * 60)
        assert limiter.allow(f"client-{window}")[0] is True

    assert len(limiter._requests) == 1
    assert ("client-99", 99) in limiter._requests


def test_download_cli_feeds_default_web_app_full_catalog_and_cached_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_bytes = Path("tests/fixtures/sky_chart/hyg-valid.csv").read_bytes()
    archive = gzip.compress(csv_bytes, mtime=0)
    packaged_source = sky_chart_catalog_module.load_hyg_source()
    source = HygSource(
        url=packaged_source.url,
        asset_name="hygdata_v41.csv.gz",
        version="4.1",
        license=packaged_source.license,
        compressed_sha256=sha256(archive).hexdigest(),
    )
    catalog_cache_dir = tmp_path / "catalog-cache"
    monkeypatch.setattr(cli_module, "load_hyg_source", lambda: source)
    monkeypatch.setattr(
        cli_module,
        "HttpCatalogFetcher",
        lambda: BytesCatalogFetcher(archive),
    )
    assert (
        cli_module.main(
            [
                "sky-chart",
                "--download-catalog",
                "--catalog-cache-dir",
                str(catalog_cache_dir),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["downloaded"] is True

    target_name = "NGC 1976"
    target_cache_dir = tmp_path / "target-cache"
    cached_target = ResolvedTarget(
        input_name=target_name,
        query_name=target_name,
        canonical_name="Orion Nebula",
        ra_deg=83.822083,
        dec_deg=-5.391111,
        object_type="HII",
        aliases=["M 42"],
        source=TargetSource(
            database="SIMBAD",
            service_url="https://simbad.example.test",
            accessed_at="2026-01-01T00:00:00Z",
            from_cache=False,
        ),
    )
    cached_path = target_cache_path(target_cache_dir, target_name)
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(cached_target.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr(sky_chart_catalog_module, "load_hyg_source", lambda: source)
    monkeypatch.setenv("STARSKILL_TARGET_CACHE_DIR", str(target_cache_dir))
    monkeypatch.setenv("STARSKILL_RUNS_DIR", str(tmp_path / "runs"))

    class ForbiddenNetworkBackend:
        def __init__(self) -> None:
            raise AssertionError("production composition attempted a live target lookup")

    monkeypatch.setattr(target_resolver_module, "SimbadBackend", ForbiddenNetworkBackend)
    client = TestClient(
        web_api_module.default_web_app(catalog_cache_dir=catalog_cache_dir)
    )
    payload = valid_sky_chart_payload(catalog_mode="full")
    payload["target"] = {"mode": "name", "name": target_name}

    response = client.post("/v1/sky-chart/render", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["catalog_mode_used"] == "full"
    assert result["catalog_status"] == "available"
    metadata = client.get(result["json_url"])
    assert metadata.status_code == 200
    assert metadata.json()["request"]["target"]["resolved"]["label"] == "Orion Nebula"
    assert str(tmp_path) not in response.text
    assert str(tmp_path) not in metadata.text


def test_default_web_app_maps_target_backend_failure_to_stable_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STARSKILL_TARGET_CACHE_DIR", str(tmp_path / "target-cache"))
    monkeypatch.setenv("STARSKILL_RUNS_DIR", str(tmp_path / "runs"))

    class FailingBackend:
        service_url = "https://simbad.example.test"

        def query_object(self, _query_name: str) -> object:
            raise RuntimeError("resolver failed at /private/target-cache")

    monkeypatch.setattr(target_resolver_module, "SimbadBackend", FailingBackend)
    client = TestClient(
        web_api_module.default_web_app(catalog_cache_dir=tmp_path / "catalog-cache")
    )
    payload = valid_sky_chart_payload()
    payload["target"] = {"mode": "name", "name": "NGC 1976"}

    response = client.post("/v1/sky-chart/render", json=payload)

    assert response.status_code == 200
    assert response.json()["warnings"] == ["target_resolution_unavailable"]
    assert "private" not in response.text


def test_run_web_server_rejects_invalid_port_before_constructing_app() -> None:
    called = False

    def app_factory() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(ValueError, match="1024..65535"):
        web_api_module.run_web_server(1023, False, web_app_factory=app_factory)
    assert called is False


def test_run_web_server_hard_binds_loopback_and_opens_after_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = Event()
    observed: dict[str, object] = {}

    def fake_config(app: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return SimpleNamespace(app=app)

    class FakeServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.started = False

        def run(self) -> None:
            self.started = True
            assert opened.wait(timeout=1)

    def health_get(url: str) -> int:
        observed["health_url"] = url
        return 200

    def browser_open(url: str) -> bool:
        observed["browser_url"] = url
        opened.set()
        return True

    monkeypatch.setattr(web_api_module.uvicorn, "Config", fake_config)
    monkeypatch.setattr(web_api_module.uvicorn, "Server", FakeServer)

    web_api_module.run_web_server(
        8123,
        True,
        web_app_factory=lambda: object(),
        health_get=health_get,
        browser_open=browser_open,
    )

    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8123
    assert observed["proxy_headers"] is False
    assert observed["health_url"] == "http://127.0.0.1:8123/healthz"
    assert observed["browser_url"] == "http://127.0.0.1:8123/"


def test_bind_failure_propagates_without_opening_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    class FailingServer:
        started = False

        def __init__(self, config: object) -> None:
            self.config = config

        def run(self) -> None:
            raise OSError("address already in use")

    monkeypatch.setattr(web_api_module.uvicorn, "Server", FailingServer)

    def browser_open(url: str) -> bool:
        nonlocal opened
        opened = True
        return True

    with pytest.raises(OSError, match="address already in use"):
        web_api_module.run_web_server(
            8123,
            True,
            web_app_factory=lambda: object(),
            health_get=lambda _url: 200,
            browser_open=browser_open,
        )
    assert opened is False


def test_bind_failure_without_open_does_not_print_success_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingServer:
        started = False

        def __init__(self, config: object) -> None:
            self.config = config

        def run(self) -> None:
            raise OSError("address already in use")

    monkeypatch.setattr(web_api_module.uvicorn, "Server", FailingServer)

    with pytest.raises(OSError, match="address already in use"):
        web_api_module.run_web_server(
            8123,
            False,
            web_app_factory=lambda: object(),
        )

    assert capsys.readouterr().out == ""


def test_non_browser_url_is_announced_only_after_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    announced = Event()
    events: list[str] = []

    class HealthyServer:
        started = False

        def __init__(self, config: object) -> None:
            self.config = config

        def run(self) -> None:
            events.append("server-running")
            self.started = True
            assert announced.wait(timeout=1)

    def health_get(_url: str) -> int:
        events.append("health-ok")
        return 200

    def record_print(value: str) -> None:
        events.append(f"printed:{value}")
        announced.set()

    monkeypatch.setattr(web_api_module.uvicorn, "Server", HealthyServer)
    monkeypatch.setattr("builtins.print", record_print)

    web_api_module.run_web_server(
        8123,
        False,
        web_app_factory=lambda: object(),
        health_get=health_get,
    )

    assert events == [
        "server-running",
        "health-ok",
        "printed:http://127.0.0.1:8123/",
    ]


def test_main_parses_only_port_and_never_opens_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        web_api_module,
        "run_web_server",
        lambda port, open_browser: called.append((port, open_browser)),
    )

    web_api_module.main(["--port", "9123"])

    assert called == [(9123, False)]
    with pytest.raises(SystemExit):
        web_api_module.main(["--host", "0.0.0.0"])
