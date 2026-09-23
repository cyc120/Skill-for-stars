from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image, ImageChops
import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Circle, Wedge
import pytest

import starskill.sky_chart as sky_chart_module
from starskill.schemas import SkyChartExportMetadata, SkyChartRequest
from starskill.sky_chart import (
    RenderStore,
    RenderedSkyChart,
    SkyChartRenderer,
    SkyChartService,
    sort_stars_dim_to_bright,
)
from starskill.sky_chart_catalog import (
    BundledCatalog,
    CatalogMetadata,
    CatalogSelection,
    CatalogStar,
    ConstellationSegment,
    FullCatalog,
)
from starskill.sky_chart_targets import SkyChartTargetResolver
from starskill.target_resolver import (
    InvalidTargetNameError,
    TargetNotFoundError,
    TargetServiceError,
)


FIXED_REQUEST = SkyChartRequest.model_validate(
    {
        "observer": {
            "location_name": "Beijing",
            "longitude": 116.4074,
            "latitude": 39.9042,
            "timezone": "Asia/Shanghai",
        },
        "timestamp_local": "2026-01-10T20:00:00+08:00",
        "target": {
            "mode": "coordinates",
            "ra_deg": 83.822083,
            "dec_deg": -5.391111,
        },
        "catalog_mode": "bundled",
    }
)
FIXED_CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


class EmptyFullCache:
    def load_valid(self) -> None:
        return None


class FixedFullCache:
    def __init__(self, catalog: FullCatalog) -> None:
        self.catalog = catalog
        self.calls = 0

    def load_valid(self) -> FullCatalog:
        self.calls += 1
        return self.catalog


@pytest.fixture(scope="module")
def service() -> SkyChartService:
    return SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_resolver=SkyChartTargetResolver(lambda _name: None),
        utc_clock=lambda: FIXED_CREATED_AT,
    )


@pytest.fixture(scope="module")
def fixed_chart(service: SkyChartService):
    return service.render(FIXED_REQUEST)


def chart_with_png_bytes(template: RenderedSkyChart, png_bytes: bytes) -> RenderedSkyChart:
    metadata = template.metadata.model_copy(
        update={
            "render": template.metadata.render.model_copy(
                update={"png_sha256": sha256(png_bytes).hexdigest()}
            )
        }
    )
    return RenderedSkyChart(
        png_bytes=png_bytes,
        metadata=metadata,
        catalog_mode_used=template.catalog_mode_used,
        catalog_status=template.catalog_status,
    )


def stored_byte_size(chart: RenderedSkyChart) -> int:
    metadata = chart.metadata.model_copy(update={"render_id": "A" * 32})
    metadata_json = metadata.model_dump_json(
        exclude_none=False,
        by_alias=True,
    ).encode("utf-8")
    return len(chart.png_bytes) + len(metadata_json)


def test_render_has_expected_layer_order_and_linked_png_digest(fixed_chart) -> None:
    assert fixed_chart.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert fixed_chart.metadata.render.layer_order == [
        "background",
        "horizon_grid",
        "constellations",
        "stars",
        "moon",
        "planets",
        "target",
        "footer",
    ]
    assert fixed_chart.metadata.render.png_sha256 == sha256(
        fixed_chart.png_bytes
    ).hexdigest()
    assert fixed_chart.metadata.catalog.constellation_segments.sha256 != (
        fixed_chart.metadata.catalog.sha256
    )
    assert fixed_chart.metadata.calculation.horizontal_frame == "AltAz"
    assert fixed_chart.metadata.calculation.atmospheric_refraction is False


def test_png_is_exact_rgb_canvas_and_nonblank(fixed_chart) -> None:
    image = Image.open(BytesIO(fixed_chart.png_bytes))
    assert image.size == (2400, 2400)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert ImageChops.difference(image, Image.new("RGB", image.size)).getbbox()


def test_sky_disk_fills_the_canvas(fixed_chart) -> None:
    # A circle in a 4:3 canvas wastes a quarter of its width, so this guards the
    # shape of the layout rather than the pixel count. Masking the exact background
    # fill excludes the cardinal labels, constellation lines and stars near the rim.
    image = Image.open(BytesIO(fixed_chart.png_bytes)).convert("RGB")
    pixels = np.asarray(image)
    disk = np.all(pixels == (5, 7, 12), axis=2)
    rows = np.flatnonzero(disk.any(axis=1))
    cols = np.flatnonzero(disk.any(axis=0))
    diameter = max(cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1)

    assert diameter >= 0.86 * image.width
    assert diameter >= 0.86 * image.height
    # Centred horizontally; the vertical centre sits higher to leave the footer band.
    assert abs((cols[0] + cols[-1]) / 2 - image.width / 2) <= 4
    assert pixels[0, 0].tolist() == [0, 0, 0]
    assert pixels[-1, -1].tolist() == [0, 0, 0]


def test_label_scale_matches_the_live_axes_transform() -> None:
    # The label collision model converts points to data units through this scale.
    # Deriving it from the requested box rather than the aspect-adjusted one made it
    # 17% too small once before, which let overlapping labels go undetected.
    figure = Figure(
        figsize=(
            sky_chart_module.CANVAS_WIDTH_PX / sky_chart_module.CANVAS_DPI,
            sky_chart_module.CANVAS_HEIGHT_PX / sky_chart_module.CANVAS_DPI,
        ),
        dpi=sky_chart_module.CANVAS_DPI,
    )
    FigureCanvasAgg(figure)
    axes = figure.add_axes(sky_chart_module.PLOT_BOX)
    axes.set_xlim(-sky_chart_module.AXIS_LIMIT, sky_chart_module.AXIS_LIMIT)
    axes.set_ylim(-sky_chart_module.AXIS_LIMIT, sky_chart_module.AXIS_LIMIT)
    axes.set_aspect("equal")
    try:
        figure.canvas.draw()
        inverse = axes.transData.inverted()
        (_, y0), (_, y1) = inverse.transform((0, 0)), inverse.transform((0, 1))
        measured = abs(y1 - y0)

        assert measured == pytest.approx(
            sky_chart_module._DATA_UNITS_PER_PIXEL, rel=0.01
        )
    finally:
        figure.clear()


def test_text_sizes_clear_the_screen_legibility_floor() -> None:
    # A glyph is worth ``pt * (DPI/72) * (display_side / CANVAS_WIDTH_PX)`` CSS
    # pixels; DPI and pixel count cancel, so only the canvas-pixel size matters.
    display_side = 770.0  # fit-mode chart side on a 1440x900 viewport
    scale = display_side / sky_chart_module.CANVAS_WIDTH_PX
    for canvas_px, floor_css_px in (
        (sky_chart_module._FOOTER_FONT_PX, 12.0),
        (sky_chart_module._OBJECT_LABEL_FONT_PX, 11.0),
        (sky_chart_module._TARGET_LABEL_FONT_PX, 11.0),
        (sky_chart_module._CARDINAL_FONT_PX, 12.0),
    ):
        assert canvas_px * scale >= floor_css_px


def test_fixed_render_png_bytes_are_deterministic(service, fixed_chart) -> None:
    repeated = service.render(FIXED_REQUEST)
    assert repeated.png_bytes == fixed_chart.png_bytes
    assert repeated.metadata.render.png_sha256 == fixed_chart.metadata.render.png_sha256


CJK_LABEL_SAMPLE = "木星 天王星 北京"


def render_label(families: list[str]) -> bytes:
    with sky_chart_module.deterministic_astropy_matplotlib():
        matplotlib.rcParams["font.family"] = families
        figure = Figure(figsize=(3, 0.6), facecolor="#000000")
        FigureCanvasAgg(figure)
        figure.text(
            0.5, 0.5, CJK_LABEL_SAMPLE, color="#ffffff", fontsize=12, ha="center", va="center"
        )
        buffer = BytesIO()
        with warnings.catch_warnings():
            # The DejaVu-only control is expected to miss these glyphs.
            warnings.simplefilter("ignore", UserWarning)
            figure.savefig(buffer, format="png", facecolor="#000000")
        return buffer.getvalue()


def test_font_family_keeps_dejavu_first_and_appends_cjk_fallbacks() -> None:
    with sky_chart_module.deterministic_astropy_matplotlib():
        families = list(matplotlib.rcParams["font.family"])
    assert families[0] == "DejaVu Sans"
    assert families[1:] == list(sky_chart_module.cjk_font_fallbacks())


def test_resolved_cjk_fallbacks_cover_the_glyphs_the_chart_draws() -> None:
    fallbacks = sky_chart_module.cjk_font_fallbacks()
    if not fallbacks:
        pytest.skip("host has no CJK-capable font installed")
    for character in CJK_LABEL_SAMPLE.replace(" ", ""):
        assert any(
            FT2Font(
                font_manager.findfont(
                    font_manager.FontProperties(family=name), fallback_to_default=False
                )
            ).get_char_index(ord(character))
            for name in fallbacks
        ), character


def test_cjk_labels_render_differently_from_dejavu_only() -> None:
    fallbacks = sky_chart_module.cjk_font_fallbacks()
    if not fallbacks:
        pytest.skip("host has no CJK-capable font installed")
    assert render_label(["DejaVu Sans", *fallbacks]) != render_label(["DejaVu Sans"])


def test_host_without_cjk_fonts_keeps_the_dejavu_only_family(monkeypatch) -> None:
    monkeypatch.setattr(sky_chart_module.font_manager.fontManager, "ttflist", [])
    sky_chart_module.cjk_font_fallbacks.cache_clear()
    try:
        assert sky_chart_module.cjk_font_fallbacks() == ()
        with sky_chart_module.deterministic_astropy_matplotlib():
            assert list(matplotlib.rcParams["font.family"]) == ["DejaVu Sans"]
    finally:
        sky_chart_module.cjk_font_fallbacks.cache_clear()


def test_invisible_object_is_recorded_but_not_drawn(fixed_chart) -> None:
    objects = [fixed_chart.metadata.objects.moon, *fixed_chart.metadata.objects.planets]
    assert all(item.drawn is item.visible for item in objects)
    assert any(not item.visible for item in objects)


def test_moon_and_seven_planets_have_complete_metadata(fixed_chart) -> None:
    moon = fixed_chart.metadata.objects.moon
    assert moon.icrs is not None
    assert moon.illumination_fraction is not None
    assert 0 <= moon.illumination_fraction <= 1
    assert [planet.label for planet in fixed_chart.metadata.objects.planets] == [
        "Mercury / 水星",
        "Venus / 金星",
        "Mars / 火星",
        "Jupiter / 木星",
        "Saturn / 土星",
        "Uranus / 天王星",
        "Neptune / 海王星",
    ]
    assert all(planet.icrs is not None for planet in fixed_chart.metadata.objects.planets)
    assert all("sun" not in planet.label.casefold() for planet in fixed_chart.metadata.objects.planets)


def test_moon_patch_coverage_tracks_illumination_and_stays_below_horizon_hidden(
    fixed_chart,
) -> None:
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    visible_moon = fixed_chart.metadata.objects.moon.model_copy(
        update={
            "altaz": fixed_chart.metadata.objects.moon.altaz.model_copy(
                update={"altitude_deg": 45.0, "azimuth_deg": 180.0}
            ),
            "visible": True,
            "drawn": True,
            "illumination_fraction": 0.25,
        }
    )
    labels = sky_chart_module._LabelPlacer(axes)
    try:
        SkyChartRenderer._draw_moon(axes, visible_moon, labels)
        labels.draw()
        assert [type(patch) for patch in axes.patches] == [Circle, Wedge, Circle]
        illuminated = axes.patches[1]
        assert isinstance(illuminated, Wedge)
        assert illuminated.theta2 - illuminated.theta1 == pytest.approx(90.0)

        hidden_moon = visible_moon.model_copy(
            update={
                "altaz": visible_moon.altaz.model_copy(
                    update={"altitude_deg": -1.0}
                ),
                "visible": False,
                "drawn": True,
                "illumination_fraction": 0.75,
            }
        )
        SkyChartRenderer._draw_moon(axes, hidden_moon, labels)
        labels.draw()
        assert len(axes.patches) == 3
        assert len(axes.texts) == 1
    finally:
        figure.clear()


def label_box(text) -> tuple[float, float, float, float]:
    width, height = sky_chart_module._label_extent(text.get_text(), text.get_fontsize())
    x, y = text.get_position()
    return (x, y, x + width, y + height)


def test_nearby_labels_are_offset_so_they_do_not_overprint() -> None:
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    labels = sky_chart_module._LabelPlacer(axes)
    try:
        labels.add(0.0, 0.0, "Jupiter / 木星", color="#fff", fontsize=6.5, zorder=5)
        labels.add(0.01, 0.01, "Neptune / 海王星", color="#fff", fontsize=6.5, zorder=5)
        labels.add(0.02, -0.01, "Saturn / 土星", color="#fff", fontsize=6.5, zorder=5)
        labels.draw()

        boxes = [label_box(text) for text in axes.texts]
        for index, box in enumerate(boxes):
            for other in boxes[index + 1 :]:
                assert not sky_chart_module._boxes_overlap(box, other)
    finally:
        figure.clear()


def test_a_lone_label_keeps_the_default_offset() -> None:
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    labels = sky_chart_module._LabelPlacer(axes)
    try:
        labels.add(0.0, 0.0, "M42", color="#fff", fontsize=8, zorder=6)
        labels.add(0.8, 0.8, "Jupiter / 木星", color="#fff", fontsize=6.5, zorder=5)
        labels.draw()

        assert axes.texts[1].get_position() == pytest.approx((0.825, 0.825))
    finally:
        figure.clear()


def test_renderer_clears_figure_when_png_encoding_fails(monkeypatch) -> None:
    cleared_figures = []
    original_clear = Figure.clear

    def tracked_clear(figure, *args, **kwargs):
        cleared_figures.append(figure)
        return original_clear(figure, *args, **kwargs)

    def fail_encoding(_figure):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(Figure, "clear", tracked_clear)
    monkeypatch.setattr(
        SkyChartRenderer,
        "_save_rgb_png",
        staticmethod(fail_encoding),
    )

    with pytest.raises(RuntimeError, match="encoding failed"):
        SkyChartService(
            full_catalog_cache=EmptyFullCache(),
            target_resolver=SkyChartTargetResolver(lambda _name: None),
            utc_clock=lambda: FIXED_CREATED_AT,
        ).render(FIXED_REQUEST)

    assert cleared_figures
    assert len({id(figure) for figure in cleared_figures}) == 1
    assert cleared_figures[-1].axes == []


def test_star_magnitude_order_is_dim_to_bright() -> None:
    stars = (
        CatalogStar("bright", "Bright", 0, 0, -1),
        CatalogStar("dim", "Dim", 0, 0, 5),
        CatalogStar("middle", "Middle", 0, 0, 2),
    )
    assert [star.star_id for star in sort_stars_dim_to_bright(stars)] == [
        "dim",
        "middle",
        "bright",
    ]


def test_constellation_segment_requires_both_endpoints_above_horizon() -> None:
    catalog = BundledCatalog(
        stars=(
            CatalogStar("north", "North", 0, 90, 1),
            CatalogStar("south", "South", 0, -90, 1),
        ),
        segments=(ConstellationSegment("Test", "north", "south"),),
        metadata=CatalogMetadata("test", "1", "https://example.test", "CC0", "a" * 64),
        segment_metadata=CatalogMetadata(
            "test", "1", "https://example.test", "CC0", "b" * 64
        ),
    )
    request = FIXED_REQUEST.model_copy(
        update={
            "observer": FIXED_REQUEST.observer.model_copy(
                update={"longitude": 0.0, "latitude": 90.0, "timezone": "UTC"}
            ),
            "timestamp_local": datetime(2026, 1, 10, 12, tzinfo=timezone.utc),
        }
    )
    chart = SkyChartRenderer(utc_clock=lambda: FIXED_CREATED_AT).render(
        request,
        CatalogSelection(
            "bundled",
            "available",
            catalog,
            catalog.segment_metadata,
            catalog.segments,
        ),
        None,
    )
    assert chart.metadata.objects.stars_drawn == 1
    assert chart.metadata.objects.constellation_segments_drawn == 0


@pytest.mark.parametrize(
    ("error", "warning"),
    [
        (InvalidTargetNameError("private detail"), "target_unresolved"),
        (TargetNotFoundError("private detail"), "target_unresolved"),
        (TargetServiceError("private detail"), "target_resolution_unavailable"),
    ],
)
def test_target_resolution_failures_become_stable_warning_only(
    error: Exception, warning: str
) -> None:
    request = FIXED_REQUEST.model_copy(
        update={
            "target": FIXED_REQUEST.target.model_copy(
                update={"mode": "name", "name": "Example", "ra_deg": None, "dec_deg": None}
            )
        }
    )
    resolver = SkyChartTargetResolver(
        lambda _name: (_ for _ in ()).throw(error)
    )
    chart = SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_resolver=resolver,
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(request)
    serialized = chart.metadata.model_dump_json(exclude_none=False, by_alias=True)

    assert chart.metadata.objects.target is None
    assert chart.metadata.warnings == [warning]
    assert "private detail" not in serialized


@pytest.mark.parametrize("body", ["Pluto", "Ceres"])
def test_unsupported_solar_system_target_becomes_unresolved_warning(
    body: str,
) -> None:
    request = FIXED_REQUEST.model_copy(
        update={
            "target": FIXED_REQUEST.target.model_copy(
                update={"mode": "name", "name": body, "ra_deg": None, "dec_deg": None}
            )
        }
    )

    def fail_if_called(_name: str):
        raise AssertionError("external resolver must not receive solar-system body")

    chart = SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_resolver=SkyChartTargetResolver(fail_if_called),
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(request)

    assert chart.metadata.objects.target is None
    assert chart.metadata.warnings == ["target_unresolved"]


def test_target_cache_write_failure_becomes_stable_nonfatal_warning(
    tmp_path: Path,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    class StaticBackend:
        service_url = "https://simbad.example.test"

        def query_object(self, _query_name: str) -> dict[str, object]:
            return {
                "canonical_name": "Example",
                "ra_deg": 1.0,
                "dec_deg": 2.0,
                "object_type": "Star",
                "aliases": [],
            }

    request = FIXED_REQUEST.model_copy(
        update={
            "target": FIXED_REQUEST.target.model_copy(
                update={"mode": "name", "name": "Example", "ra_deg": None, "dec_deg": None}
            )
        }
    )

    chart = SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_cache_dir=blocked_parent / "cache",
        target_backend=StaticBackend(),
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(request)

    assert chart.metadata.objects.target is None
    assert chart.metadata.warnings == ["target_resolution_unavailable"]
    assert str(blocked_parent) not in chart.metadata.model_dump_json()


def test_none_target_resolution_becomes_target_unresolved() -> None:
    request = FIXED_REQUEST.model_copy(
        update={
            "target": FIXED_REQUEST.target.model_copy(
                update={"mode": "name", "name": "Example", "ra_deg": None, "dec_deg": None}
            )
        }
    )
    chart = SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_resolver=SkyChartTargetResolver(lambda _name: None),
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(request)
    assert chart.metadata.objects.target is None
    assert chart.metadata.warnings == ["target_unresolved"]


def test_auto_catalog_degradation_is_in_status_warning_and_footer() -> None:
    chart = SkyChartService(
        full_catalog_cache=EmptyFullCache(),
        target_resolver=SkyChartTargetResolver(lambda _name: None),
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(FIXED_REQUEST.model_copy(update={"catalog_mode": "auto"}))
    assert chart.catalog_mode_used == "bundled"
    assert chart.catalog_status == "degraded"
    assert chart.metadata.catalog.status == "degraded"
    assert chart.metadata.warnings == ["catalog_degraded"]


def test_full_catalog_selection_is_cache_only_and_uses_bundled_segments() -> None:
    full = FullCatalog(
        stars=(CatalogStar("full-1", "Full", 0, 90, 1),),
        metadata=CatalogMetadata("full", "1", "https://example.test/full", "CC0", "b" * 64),
        row_count=100_001,
    )
    cache = FixedFullCache(full)
    chart = SkyChartService(
        full_catalog_cache=cache,
        target_resolver=SkyChartTargetResolver(lambda _name: None),
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(FIXED_REQUEST.model_copy(update={"catalog_mode": "full"}))
    assert cache.calls == 1
    assert chart.catalog_mode_used == "full"
    assert chart.catalog_status == "available"
    assert chart.metadata.catalog.dataset_id == "full"


def test_full_catalog_draws_segments_from_bundled_endpoint_stars() -> None:
    bundled = BundledCatalog(
        stars=(
            CatalogStar("a", "A", 0, 90, 1),
            CatalogStar("b", "B", 90, 90, 1),
        ),
        segments=(ConstellationSegment("Test", "a", "b"),),
        metadata=CatalogMetadata("bundled", "1", "https://example.test/b", "CC0", "a" * 64),
        segment_metadata=CatalogMetadata(
            "bundled", "1", "https://example.test/b", "CC0", "c" * 64
        ),
    )
    full = FullCatalog(
        stars=(CatalogStar("hyg-1", "Full", 180, 90, 1),),
        metadata=CatalogMetadata("full", "1", "https://example.test/f", "CC0", "b" * 64),
        row_count=100_001,
    )
    request = FIXED_REQUEST.model_copy(
        update={
            "catalog_mode": "full",
            "observer": FIXED_REQUEST.observer.model_copy(
                update={"longitude": 0.0, "latitude": 90.0, "timezone": "UTC"}
            ),
            "timestamp_local": datetime(2026, 1, 10, 12, tzinfo=timezone.utc),
        }
    )
    chart = SkyChartService(
        bundled_catalog=bundled,
        full_catalog_cache=FixedFullCache(full),
        target_resolver=SkyChartTargetResolver(lambda _name: None),
        utc_clock=lambda: FIXED_CREATED_AT,
    ).render(request)
    assert chart.metadata.objects.stars_drawn == 1
    assert chart.metadata.objects.constellation_segments_drawn == 1


def test_serialized_coordinates_have_exactly_six_decimal_places(fixed_chart) -> None:
    serialized = fixed_chart.metadata.model_dump_json(exclude_none=False, by_alias=True)
    coordinate_tokens = re.findall(
        r'"(?:longitude|latitude|ra_deg|dec_deg|altitude_deg|azimuth_deg)":(-?\d+\.\d+)',
        serialized,
    )
    assert coordinate_tokens
    assert all(len(token.rsplit(".", 1)[1]) == 6 for token in coordinate_tokens)


def test_store_serializes_once_and_returns_exact_export_bytes(fixed_chart, monkeypatch) -> None:
    calls = 0
    original = SkyChartExportMetadata.model_dump_json

    def counted(self, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs == {"exclude_none": False, "by_alias": True}
        return original(self, **kwargs)

    monkeypatch.setattr(SkyChartExportMetadata, "model_dump_json", counted)
    store = RenderStore()
    render_id = store.put(fixed_chart)
    stored = store.get(render_id)

    assert calls == 1
    assert stored is not None
    assert stored.metadata.render_id == render_id
    assert stored.metadata_json_bytes is not None
    assert json.loads(stored.metadata_json_bytes)["render_id"] == render_id
    assert json.loads(stored.metadata_json_bytes)["render"]["png_sha256"] == sha256(
        stored.png_bytes
    ).hexdigest()


def test_render_store_rejects_png_metadata_digest_mismatch(fixed_chart) -> None:
    mismatched = RenderedSkyChart(
        png_bytes=b"different bytes",
        metadata=fixed_chart.metadata,
        catalog_mode_used=fixed_chart.catalog_mode_used,
        catalog_status=fixed_chart.catalog_status,
    )

    with pytest.raises(ValueError, match="metadata does not match"):
        RenderStore().put(mismatched)


def test_render_store_rejects_matching_digest_non_png_bytes(fixed_chart) -> None:
    invalid_png = chart_with_png_bytes(fixed_chart, b"not a png")

    with pytest.raises(ValueError, match="valid 2400x2400 RGB PNG"):
        RenderStore().put(invalid_png)


def test_render_store_expires_and_malformed_ids_match_missing(fixed_chart) -> None:
    now = [0.0]
    store = RenderStore(
        ttl_seconds=900,
        max_records=2,
        max_bytes=stored_byte_size(fixed_chart),
        monotonic_clock=lambda: now[0],
    )
    first = store.put(fixed_chart)
    assert re.fullmatch(r"[A-Za-z0-9_-]{32}", first)
    assert store.get("not/valid") is None
    assert store.get("missing_but_valid") is None
    now[0] = 901.0
    assert store.get(first) is None


def test_render_store_retries_malformed_generated_id(fixed_chart, monkeypatch) -> None:
    generated = iter(["not/url-safe", "A" * 32])
    monkeypatch.setattr(
        sky_chart_module.secrets,
        "token_urlsafe",
        lambda _bytes: next(generated),
    )
    store = RenderStore(max_bytes=stored_byte_size(fixed_chart))
    assert store.put(fixed_chart) == "A" * 32


def test_render_store_purges_before_malformed_get(fixed_chart) -> None:
    now = [0.0]
    store = RenderStore(
        ttl_seconds=1,
        max_bytes=stored_byte_size(fixed_chart),
        monotonic_clock=lambda: now[0],
    )
    render_id = store.put(fixed_chart)
    now[0] = 2.0
    assert store.get("bad/id") is None
    now[0] = 0.0
    assert store.get(render_id) is None


def test_render_store_evicts_at_default_record_capacity(fixed_chart) -> None:
    record_size = stored_byte_size(fixed_chart)
    store = RenderStore(max_bytes=21 * record_size)
    assert store.max_records == 20
    ids = [store.put(fixed_chart) for _index in range(21)]
    assert store.get(ids[0]) is None
    assert all(store.get(render_id) is not None for render_id in ids[1:])


def test_render_store_evicts_at_byte_capacity_and_uses_50_mib_default(
    fixed_chart,
) -> None:
    assert RenderStore().max_bytes == 50 * 1024 * 1024
    one_record_size = stored_byte_size(fixed_chart)
    store = RenderStore(max_records=20, max_bytes=2 * one_record_size - 1)
    first = store.put(fixed_chart)
    second = store.put(fixed_chart)
    assert store.get(first) is None
    assert store.get(second) is not None


def test_render_store_evicts_earliest_expiry_then_insertion_order_and_clears(
    fixed_chart,
) -> None:
    now = [0.0]
    record_size = stored_byte_size(fixed_chart)
    store = RenderStore(
        ttl_seconds=10,
        max_records=2,
        max_bytes=2 * record_size,
        monotonic_clock=lambda: now[0],
    )
    first = store.put(fixed_chart)
    second = store.put(fixed_chart)
    now[0] = 1.0
    third = store.put(fixed_chart)
    assert store.get(first) is None
    assert store.get(second) is not None
    assert store.get(third) is not None
    store.clear()
    assert store.get(second) is None
    assert store.get(third) is None
