"""Deterministic, local sky-chart rendering and bounded in-memory exports."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
import platform
import re
import secrets
import struct
import threading
import warnings
import zlib
from typing import Any, Callable, Iterator, Protocol, Sequence

from astropy import units as u
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    SkyCoord,
    get_body,
    solar_system_ephemeris,
)
from astropy.time import Time
import astropy
import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Circle, Wedge
from matplotlib.textpath import TextPath
import numpy as np

import starskill.target_resolver as target_resolver_module
from starskill.astropy_offline import offline_iers
from starskill.schemas import (
    SkyChartAltAzCoordinates,
    SkyChartCalculationMetadata,
    SkyChartCatalogMetadata,
    SkyChartCatalogSourceMetadata,
    SkyChartDependenciesMetadata,
    SkyChartExportMetadata,
    SkyChartExportRequest,
    SkyChartExportTarget,
    SkyChartIcrsCoordinates,
    SkyChartObject,
    SkyChartObjectsMetadata,
    SkyChartRenderMetadata,
    SkyChartRequest,
)
from starskill.sky_chart_catalog import (
    BundledCatalog,
    CatalogSelection,
    CatalogStar,
    ConstellationSegment,
    FullCatalog,
    load_bundled_catalog,
    select_catalog,
)
from starskill.sky_chart_targets import ResolvedSkyTarget, SkyChartTargetResolver
from starskill.target_references import UnsupportedSolarSystemBodyError
from starskill.target_resolver import (
    InvalidTargetNameError,
    TargetBackend,
    TargetNotFoundError,
    TargetServiceError,
    resolve_target,
)


# The sky chart is a circle, so the canvas is square: a 4:3 canvas wasted a quarter
# of its width once ``set_aspect("equal")`` squared the axes box. The raster is
# always exactly CANVAS_*_PX pixels because the figure is built as
# ``figsize=(W/DPI, H/DPI), dpi=DPI``; DPI only sets how many pixels a point is
# worth (``px = pt * DPI / 72``). Raise both together to scale, not DPI alone.
CANVAS_WIDTH_PX = 2400
CANVAS_HEIGHT_PX = 2400
CANVAS_DPI = 200
# Axes box in figure fractions, and the data half-range it is drawn over. The box is
# already square, so ``set_aspect("equal")`` never has to shrink it and the disk is
# limited only by the box side. Bottom inset leaves a band for the two-line footer.
PLOT_BOX = (0.04, 0.062, 0.918, 0.918)
AXIS_LIMIT = 1.05


def pt_from_canvas_px(canvas_px: float) -> float:
    """Convert a size in canvas pixels to points at CANVAS_DPI.

    Sizes written in points silently change meaning whenever the canvas or the DPI
    moves, because the raster is fixed at CANVAS_*_PX and DPI only sets how many
    pixels a point is worth. Writing the drawing sizes in canvas pixels instead
    keeps them stable across any future canvas change.
    """
    return canvas_px * 72.0 / CANVAS_DPI


def marker_area(canvas_px: float) -> float:
    """Scatter ``s`` (points squared) for a marker that wide in canvas pixels."""
    return pt_from_canvas_px(canvas_px) ** 2


def text_width_px(text: str, fontsize_pt: float) -> float:
    """Width of ``text`` in canvas pixels using the chart's own fonts.

    The footer carries the observer's free-text place name, which the schema allows
    to run to 80 characters, so its width has to be measured rather than assumed.
    """
    if not text:
        return 0.0
    path = TextPath(
        (0.0, 0.0),
        text,
        size=fontsize_pt,
        prop=font_manager.FontProperties(family=matplotlib.rcParams["font.family"]),
    )
    return path.get_extents().width / 72.0 * CANVAS_DPI


# Text sizes are chosen for on-screen legibility, not for the canvas: a glyph is
# worth ``pt * (DPI/72) * (display_side / CANVAS_WIDTH_PX)`` CSS pixels, and at the
# fit-mode display side of roughly 770 px that is about 0.29 CSS px per canvas px.
# The footer therefore needs >= 37 canvas px to clear the ~12 CSS px floor — and it
# only fits on one line up to about 10.5 pt, so it is drawn on two lines.
_FOOTER_FONT_PX = 38.0
# The observer supplies the place name and may use up to 80 characters, so the
# footer shrinks to this floor and only then truncates.
_FOOTER_MIN_FONT_PX = 24.0
_FOOTER_MAX_WIDTH_RATIO = 0.96
_OBJECT_LABEL_FONT_PX = 35.0
_TARGET_LABEL_FONT_PX = 37.0
_CARDINAL_FONT_PX = 40.0
# Decorative sizes keep the original look, scaled by the canvas growth.
_CONSTELLATION_LINE_PX = 2.9
_HORIZON_LINE_PX = 4.3
_HORIZON_INNER_LINE_PX = 2.5
_SPOKE_LINE_PX = 1.8
_ZENITH_DOT_PX = 6.2
_MOON_OUTLINE_PX = 2.9
_PLANET_MARKER_PX = 24.0
_PLANET_EDGE_PX = 1.4
_TARGET_RING_PX = 46.0
_TARGET_CROSS_PX = 26.0
_TARGET_RING_LINE_PX = 3.9
_TARGET_CROSS_LINE_PX = 4.6
# Star markers: linear width, brightest first, floored so the faintest stay visible.
_STAR_BRIGHT_PX = 20.7
_STAR_FLOOR_PX = 6.2
_STAR_MAGNITUDE_FALLOFF_PX = 3.0
LAYER_ORDER = [
    "background",
    "horizon_grid",
    "constellations",
    "stars",
    "moon",
    "planets",
    "target",
    "footer",
]
PLANETS = (
    ("mercury", "Mercury / 水星", "#b8aaa0"),
    ("venus", "Venus / 金星", "#f2d28b"),
    ("mars", "Mars / 火星", "#d96c4b"),
    ("jupiter", "Jupiter / 木星", "#d7bd9a"),
    ("saturn", "Saturn / 土星", "#d8c486"),
    ("uranus", "Uranus / 天王星", "#86d4d8"),
    ("neptune", "Neptune / 海王星", "#6688d8"),
)
_RENDER_ID_RE = re.compile(r"[A-Za-z0-9_-]{32}\Z")


class _FullCatalogCache(Protocol):
    def load_valid(self) -> FullCatalog | None: ...


class _EmptyFullCatalogCache:
    def load_valid(self) -> None:
        return None


class _LazySimbadBackend:
    service_url = target_resolver_module.SimbadBackend.service_url

    def __init__(self) -> None:
        self._backend: TargetBackend | None = None

    def query_object(self, query_name: str) -> Mapping[str, Any] | None:
        if self._backend is None:
            self._backend = target_resolver_module.SimbadBackend()
        return self._backend.query_object(query_name)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def project_altaz(altitude_deg: float, azimuth_deg: float) -> tuple[float, float]:
    radius = (90.0 - altitude_deg) / 90.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    return (
        float(radius * np.sin(azimuth_rad)),
        float(radius * np.cos(azimuth_rad)),
    )


def sort_stars_dim_to_bright(stars: Sequence[CatalogStar]) -> tuple[CatalogStar, ...]:
    """Return a stable painter's order with bright stars drawn last."""
    return tuple(sorted(stars, key=lambda star: (-star.magnitude, star.star_id)))


# PLOT_BOX is square, so this is its side; taking the min keeps it correct even if a
# future box is not square and ``set_aspect("equal")`` shrinks one direction.
_PLOT_SIDE_PX = min(PLOT_BOX[2] * CANVAS_WIDTH_PX, PLOT_BOX[3] * CANVAS_HEIGHT_PX)
_DATA_UNITS_PER_PIXEL = 2 * AXIS_LIMIT / _PLOT_SIDE_PX
# Candidate anchor offsets, tried in order until a label clears the placed ones.
_LABEL_OFFSETS = (
    (0.025, 0.025),
    (0.025, 0.075),
    (0.025, -0.045),
    (0.025, 0.125),
    (0.025, -0.095),
    (0.025, 0.175),
    (0.085, 0.025),
    (-0.085, 0.025),
)
_LABEL_PADDING = 0.006


def _label_extent(label: str, fontsize: float) -> tuple[float, float]:
    """A label's width and height in data units.

    The width is measured from the real font, not estimated per character: an
    em-per-character guess ran up to 43% narrow for wide Latin glyphs, which let the
    placer accept positions that then overprinted. The height stays the em size,
    which is conservative because glyph ink rarely reaches it.
    """
    height_px = fontsize / 72.0 * CANVAS_DPI
    return (
        text_width_px(label, fontsize) * _DATA_UNITS_PER_PIXEL,
        height_px * _DATA_UNITS_PER_PIXEL,
    )


def truncate_to_width(text: str, fontsize_pt: float, max_width_px: float) -> str:
    """Shorten ``text`` with an ellipsis until it fits ``max_width_px``."""
    if text_width_px(text, fontsize_pt) <= max_width_px:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if text_width_px(text[:middle] + "…", fontsize_pt) <= max_width_px:
            low = middle
        else:
            high = middle - 1
    return text[:low] + "…"


def fit_footer_lines(
    lines: Sequence[str], fontsize_px: float
) -> tuple[tuple[str, ...], float]:
    """Shrink, then truncate, so every footer line fits across the canvas.

    Returns the lines and the canvas-pixel font size to draw them at. Scaling the
    whole block together keeps the lines visually consistent.
    """
    max_width = CANVAS_WIDTH_PX * _FOOTER_MAX_WIDTH_RATIO
    widest = max(text_width_px(line, pt_from_canvas_px(fontsize_px)) for line in lines)
    if widest > max_width:
        fontsize_px = max(
            _FOOTER_MIN_FONT_PX, fontsize_px * max_width / widest
        )
    fontsize_pt = pt_from_canvas_px(fontsize_px)
    return (
        tuple(truncate_to_width(line, fontsize_pt, max_width) for line in lines),
        fontsize_px,
    )


def _boxes_overlap(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> bool:
    left, bottom, right, top = first
    other_left, other_bottom, other_right, other_top = second
    return not (
        right + _LABEL_PADDING <= other_left
        or left >= other_right + _LABEL_PADDING
        or top + _LABEL_PADDING <= other_bottom
        or bottom >= other_top + _LABEL_PADDING
    )


def _overlap_area(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    """Overlap of two label boxes, counting the separation padding as overlap.

    The padding is included so this ranks candidates on the same criterion
    ``_boxes_overlap`` enforces; a purely geometric area would score a
    padding-only near-miss as zero and tie with a genuinely clear candidate.
    """
    left = max(first[0], second[0] - _LABEL_PADDING)
    bottom = max(first[1], second[1] - _LABEL_PADDING)
    right = min(first[2], second[2] + _LABEL_PADDING)
    top = min(first[3], second[3] + _LABEL_PADDING)
    return max(0.0, right - left) * max(0.0, top - bottom)


def canvas_data_bounds() -> tuple[float, float, float, float]:
    """The whole canvas in data units: ``(x_min, y_min, x_max, y_max)``.

    Labels are drawn with clipping off, so nothing stops one from being painted off
    the raster. The placer needs these bounds to reject such candidates.
    """
    centre_x = PLOT_BOX[0] * CANVAS_WIDTH_PX + _PLOT_SIDE_PX / 2
    centre_y = PLOT_BOX[1] * CANVAS_HEIGHT_PX + _PLOT_SIDE_PX / 2
    scale = _PLOT_SIDE_PX / (2 * AXIS_LIMIT)
    return (
        (0.0 - centre_x) / scale,
        (0.0 - centre_y) / scale,
        (CANVAS_WIDTH_PX - centre_x) / scale,
        (CANVAS_HEIGHT_PX - centre_y) / scale,
    )


def _outside_area(box: tuple[float, float, float, float]) -> float:
    x_min, y_min, x_max, y_max = canvas_data_bounds()
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    inside = max(0.0, min(box[2], x_max) - max(box[0], x_min)) * max(
        0.0, min(box[3], y_max) - max(box[1], y_min)
    )
    return width * height - inside


class _LabelPlacer:
    """Collect sky-chart labels, then draw them so nearby ones do not overprint.

    The moon, the planets and the target all label themselves at a fixed offset,
    which collides whenever two of them sit close on the sky. Anchors are chosen
    from a fixed ladder in registration order, so the result stays deterministic.
    """

    def __init__(self, axes) -> None:
        self._axes = axes
        self._pending: list[tuple[float, float, str, str, float, float]] = []
        self._placed: list[tuple[float, float, float, float]] = []

    def add(
        self, x: float, y: float, label: str, *, color: str, fontsize: float, zorder: float
    ) -> None:
        self._pending.append((x, y, label, color, fontsize, zorder))

    def _choose_offset(
        self, x: float, y: float, width: float, height: float
    ) -> tuple[float, float]:
        """Pick the candidate that stays on the canvas and clears the placed boxes.

        Labels are not clipped by the axes, so a candidate that leaves the raster is
        worse than one that merely crowds a neighbour: it silently loses glyphs.
        Candidates are therefore ranked on (area off-canvas, area overlapping).
        """
        best: tuple[float, float] | None = None
        best_cost: tuple[float, float] | None = None
        for candidate in _LABEL_OFFSETS:
            box = (
                x + candidate[0],
                y + candidate[1],
                x + candidate[0] + width,
                y + candidate[1] + height,
            )
            # Rank on the same padded test the renderer promises to satisfy, so a
            # candidate the placer calls clear is one that really is clear.
            crowded = [placed for placed in self._placed if _boxes_overlap(box, placed)]
            outside = _outside_area(box)
            if not crowded and outside == 0.0:
                self._placed.append(box)
                return candidate
            cost = (outside, sum(_overlap_area(box, placed) for placed in crowded))
            if best_cost is None or cost < best_cost:
                best, best_cost = candidate, cost
        assert best is not None
        self._placed.append(
            (x + best[0], y + best[1], x + best[0] + width, y + best[1] + height)
        )
        return best

    def draw(self) -> None:
        pending, self._pending = self._pending, []
        for x, y, label, color, fontsize, zorder in pending:
            width, height = _label_extent(label, fontsize)
            dx, dy = self._choose_offset(x, y, width, height)
            self._axes.text(
                x + dx, y + dy, label, color=color, fontsize=fontsize, zorder=zorder
            )


# DejaVu Sans ships with matplotlib and stays the deterministic base font, but it
# carries no CJK glyphs, so the bundled bilingual planet labels and any place name
# the observer types would render as tofu boxes. matplotlib falls back glyph by
# glyph only when ``font.family`` is a list of concrete family names, so append the
# first installed CJK-capable families from this platform-ordered preference list.
_CJK_FONT_CANDIDATES = (
    "PingFang SC",  # macOS
    "Hiragino Sans GB",
    "Heiti TC",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
    "Microsoft YaHei",  # Windows
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",  # Linux
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Droid Sans Fallback",
)
_CJK_FALLBACK_LIMIT = 3
_CJK_PROBE_CHARACTER = "木"


@lru_cache(maxsize=1)
def cjk_font_fallbacks() -> tuple[str, ...]:
    """Installed CJK-capable font families, in preference order.

    Resolved once per process so repeated renders stay byte-for-byte identical.
    A machine with no CJK font installed yields an empty tuple and keeps the
    previous DejaVu-only behaviour instead of failing the render.
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    fallbacks: list[str] = []
    for name in _CJK_FONT_CANDIDATES:
        if name not in installed:
            continue
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False
            )
            covers_cjk = FT2Font(path).get_char_index(ord(_CJK_PROBE_CHARACTER)) != 0
        except (OSError, RuntimeError, ValueError):
            continue
        if not covers_cjk:
            continue
        fallbacks.append(name)
        if len(fallbacks) == _CJK_FALLBACK_LIMIT:
            break
    return tuple(fallbacks)


@contextmanager
def deterministic_astropy_matplotlib() -> Iterator[None]:
    old_rc = matplotlib.rcParams.copy()
    old_random_state = np.random.get_state()
    matplotlib.rcParams.update(
        {
            "figure.dpi": CANVAS_DPI,
            "savefig.dpi": CANVAS_DPI,
            "font.family": ["DejaVu Sans", *cjk_font_fallbacks()],
            "figure.facecolor": "#000000",
            "savefig.facecolor": "#000000",
            "savefig.transparent": False,
        }
    )
    np.random.seed(0)
    try:
        with (
            offline_iers(),
            solar_system_ephemeris.set("builtin"),
        ):
            yield
    finally:
        np.random.set_state(old_random_state)
        matplotlib.rcParams.update(old_rc)


@dataclass(frozen=True)
class RenderedSkyChart:
    png_bytes: bytes
    metadata: SkyChartExportMetadata
    catalog_mode_used: str
    catalog_status: str
    _stored_metadata_json_bytes: bytes | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def metadata_json_bytes(self) -> bytes:
        if self._stored_metadata_json_bytes is not None:
            return self._stored_metadata_json_bytes
        return _serialize_metadata(self.metadata)

    @classmethod
    def _from_store(
        cls,
        *,
        png_bytes: bytes,
        metadata: SkyChartExportMetadata,
        catalog_mode_used: str,
        catalog_status: str,
        metadata_json_bytes: bytes,
    ) -> RenderedSkyChart:
        chart = cls(png_bytes, metadata, catalog_mode_used, catalog_status)
        object.__setattr__(chart, "_stored_metadata_json_bytes", metadata_json_bytes)
        return chart


@dataclass(frozen=True)
class _RenderContext:
    time_utc: Time
    timestamp_utc: datetime
    location: EarthLocation
    frame: AltAz
    dependencies: SkyChartDependenciesMetadata


@dataclass(frozen=True)
class _StoreRecord:
    expires_at: float
    insertion_order: int
    png_bytes: bytes
    metadata_json_bytes: bytes

    @property
    def byte_size(self) -> int:
        return len(self.png_bytes) + len(self.metadata_json_bytes)


class SkyChartRenderer:
    """Render a request using one frozen coordinate and dependency context."""

    def __init__(self, *, utc_clock: Callable[[], datetime] = utc_now) -> None:
        self._utc_clock = utc_clock

    def render(
        self,
        request: SkyChartRequest,
        selection: CatalogSelection,
        resolved_target: ResolvedSkyTarget | None,
    ) -> RenderedSkyChart:
        created_at = self._utc_clock().astimezone(timezone.utc)
        with deterministic_astropy_matplotlib():
            context = self._make_context(request)
            star_horizontal = self._star_horizontal(selection.catalog.stars, context)
            star_altaz = {
                star.star_id: (float(horizontal.alt.deg), float(horizontal.az.deg))
                for star, horizontal in zip(selection.catalog.stars, star_horizontal, strict=True)
            }
            if selection.constellation_stars == selection.catalog.stars:
                constellation_altaz = star_altaz
            else:
                constellation_horizontal = self._star_horizontal(
                    selection.constellation_stars, context
                )
                constellation_altaz = {
                    star.star_id: (float(horizontal.alt.deg), float(horizontal.az.deg))
                    for star, horizontal in zip(
                        selection.constellation_stars,
                        constellation_horizontal,
                        strict=True,
                    )
                }
            moon_coord = get_body("moon", context.time_utc, context.location)
            sun_coord = get_body("sun", context.time_utc, context.location)
            moon = self._body_metadata(
                "Moon / 月球",
                moon_coord,
                context,
                illumination_fraction=self._moon_illumination(moon_coord, sun_coord),
            )
            planet_records = [
                (
                    body_name,
                    label,
                    color,
                    get_body(body_name, context.time_utc, context.location),
                )
                for body_name, label, color in PLANETS
            ]
            planets = [
                self._body_metadata(label, coordinate, context)
                for _body_name, label, _color, coordinate in planet_records
            ]
            target = self._target_metadata(resolved_target, context)

            figure = Figure(
                figsize=(CANVAS_WIDTH_PX / CANVAS_DPI, CANVAS_HEIGHT_PX / CANVAS_DPI),
                dpi=CANVAS_DPI,
                facecolor="#000000",
            )
            try:
                FigureCanvasAgg(figure)
                axes = figure.add_axes(PLOT_BOX, facecolor="#000000")
                axes.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
                axes.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
                axes.set_aspect("equal")
                axes.axis("off")

                self._draw_background(axes)
                self._draw_horizon_grid(axes)
                segments_drawn = self._draw_constellations(
                    axes, selection.constellation_segments, constellation_altaz
                )
                stars_drawn = self._draw_stars(axes, selection.catalog.stars, star_altaz)
                labels = _LabelPlacer(axes)
                self._draw_moon(axes, moon, labels)
                self._draw_planets(axes, planets, planet_records, labels)
                self._draw_target(axes, target, labels)
                labels.draw()
                self._draw_footer(figure, request, context, selection)

                png_bytes = self._save_rgb_png(figure)
            finally:
                figure.clear()
            png_digest = sha256(png_bytes).hexdigest()

        metadata = SkyChartExportMetadata(
            render_id="pending",
            created_at_utc=created_at,
            request=SkyChartExportRequest(
                observer=request.observer,
                timestamp_local=request.timestamp_local,
                timestamp_utc=context.timestamp_utc,
                target=SkyChartExportTarget(
                    mode=request.target.mode,
                    input=self._target_input(request),
                    resolved=target,
                ),
                catalog_mode_requested=request.catalog_mode,
                catalog_mode_used=selection.mode_used,
            ),
            render=SkyChartRenderMetadata(
                projection="azimuthal_equidistant_zenith",
                width_px=CANVAS_WIDTH_PX,
                height_px=CANVAS_HEIGHT_PX,
                layer_order=LAYER_ORDER,
                png_sha256=png_digest,
            ),
            objects=SkyChartObjectsMetadata(
                moon=moon,
                planets=planets,
                target=target,
                stars_drawn=stars_drawn,
                constellation_segments_drawn=segments_drawn,
            ),
            catalog=SkyChartCatalogMetadata(
                dataset_id=selection.catalog.metadata.dataset_id,
                version=selection.catalog.metadata.version,
                source_url=selection.catalog.metadata.source_url,
                license=selection.catalog.metadata.license,
                sha256=selection.catalog.metadata.sha256,
                constellation_segments=SkyChartCatalogSourceMetadata(
                    dataset_id=selection.segment_metadata.dataset_id,
                    version=selection.segment_metadata.version,
                    source_url=selection.segment_metadata.source_url,
                    license=selection.segment_metadata.license,
                    sha256=selection.segment_metadata.sha256,
                ),
                status=selection.status,
            ),
            calculation=SkyChartCalculationMetadata(),
            dependencies=context.dependencies,
            warnings=["catalog_degraded"] if selection.status == "degraded" else [],
        )
        return RenderedSkyChart(
            png_bytes=png_bytes,
            metadata=metadata,
            catalog_mode_used=selection.mode_used,
            catalog_status=selection.status,
        )

    @staticmethod
    def _make_context(request: SkyChartRequest) -> _RenderContext:
        time_utc = Time(request.timestamp_local).utc
        timestamp_utc = time_utc.to_datetime(timezone=timezone.utc)
        location = EarthLocation.from_geodetic(
            lon=request.observer.longitude * u.deg,
            lat=request.observer.latitude * u.deg,
        )
        frame = AltAz(
            obstime=time_utc,
            location=location,
            pressure=0 * u.hPa,
        )
        return _RenderContext(
            time_utc=time_utc,
            timestamp_utc=timestamp_utc,
            location=location,
            frame=frame,
            dependencies=SkyChartDependenciesMetadata(
                python=platform.python_version(),
                astropy=astropy.__version__,
                matplotlib=matplotlib.__version__,
                tzdata=_dependency_version("tzdata"),
            ),
        )

    @staticmethod
    def _star_horizontal(stars: Sequence[CatalogStar], context: _RenderContext):
        coordinates = SkyCoord(
            ra=[star.ra_deg for star in stars] * u.deg,
            dec=[star.dec_deg for star in stars] * u.deg,
            frame="icrs",
        )
        return coordinates.transform_to(context.frame)

    @staticmethod
    def _body_metadata(
        label: str,
        coordinate: SkyCoord,
        context: _RenderContext,
        *,
        illumination_fraction: float | None = None,
    ) -> SkyChartObject:
        horizontal = coordinate.transform_to(context.frame)
        icrs = coordinate.icrs
        altitude = float(horizontal.alt.deg)
        return SkyChartObject(
            label=label,
            icrs=SkyChartIcrsCoordinates(
                ra_deg=float(icrs.ra.deg) % 360,
                dec_deg=float(icrs.dec.deg),
            ),
            altaz=SkyChartAltAzCoordinates(
                altitude_deg=altitude,
                azimuth_deg=float(horizontal.az.deg) % 360,
            ),
            visible=altitude >= 0,
            drawn=altitude >= 0,
            illumination_fraction=illumination_fraction,
        )

    def _target_metadata(
        self,
        resolved: ResolvedSkyTarget | None,
        context: _RenderContext,
    ) -> SkyChartObject | None:
        if resolved is None:
            return None
        if resolved.solar_system_body is not None:
            return self._body_metadata(
                resolved.label,
                get_body(resolved.solar_system_body, context.time_utc, context.location),
                context,
            )
        assert resolved.ra_deg is not None and resolved.dec_deg is not None
        coordinate = SkyCoord(
            ra=resolved.ra_deg * u.deg,
            dec=resolved.dec_deg * u.deg,
            frame="icrs",
        )
        horizontal = coordinate.transform_to(context.frame)
        altitude = float(horizontal.alt.deg)
        return SkyChartObject(
            label=resolved.label,
            icrs=SkyChartIcrsCoordinates(
                ra_deg=resolved.ra_deg,
                dec_deg=resolved.dec_deg,
            ),
            altaz=SkyChartAltAzCoordinates(
                altitude_deg=altitude,
                azimuth_deg=float(horizontal.az.deg) % 360,
            ),
            visible=altitude >= 0,
            drawn=altitude >= 0,
        )

    @staticmethod
    def _moon_illumination(moon: SkyCoord, sun: SkyCoord) -> float:
        elongation = moon.separation(sun).rad
        return float((1.0 - math.cos(elongation)) / 2.0)

    @staticmethod
    def _draw_background(axes) -> None:
        axes.add_patch(Circle((0, 0), 1.0, facecolor="#05070c", edgecolor="none", zorder=0))

    @staticmethod
    def _draw_horizon_grid(axes) -> None:
        for altitude, radius in ((0, 1.0), (30, 2 / 3), (60, 1 / 3)):
            axes.add_patch(
                Circle(
                    (0, 0),
                    radius,
                    fill=False,
                    edgecolor="#33414d",
                    linewidth=pt_from_canvas_px(
                        _HORIZON_INNER_LINE_PX if altitude else _HORIZON_LINE_PX
                    ),
                    zorder=1,
                )
            )
        for azimuth in (0, 90, 180, 270):
            x, y = project_altaz(0, azimuth)
            axes.plot(
                [0, x],
                [0, y],
                color="#202d36",
                linewidth=pt_from_canvas_px(_SPOKE_LINE_PX),
                zorder=1,
            )
        axes.scatter(
            [0], [0], s=marker_area(_ZENITH_DOT_PX), c="#33414d", edgecolors="none", zorder=1
        )
        for label, x, y in (("N", 0, 1.025), ("E", 1.025, 0), ("S", 0, -1.025), ("W", -1.025, 0)):
            axes.text(
                x,
                y,
                label,
                color="#8ea0aa",
                fontsize=pt_from_canvas_px(_CARDINAL_FONT_PX),
                ha="center",
                va="center",
                zorder=1,
            )

    @staticmethod
    def _draw_constellations(
        axes,
        segments: Sequence[ConstellationSegment],
        star_altaz: dict[str, tuple[float, float]],
    ) -> int:
        drawn = 0
        for segment in segments:
            start = star_altaz.get(segment.start_star_id)
            end = star_altaz.get(segment.end_star_id)
            if start is None or end is None or start[0] < 0 or end[0] < 0:
                continue
            start_xy = project_altaz(*start)
            end_xy = project_altaz(*end)
            axes.plot(
                [start_xy[0], end_xy[0]],
                [start_xy[1], end_xy[1]],
                color="#40576d",
                linewidth=pt_from_canvas_px(_CONSTELLATION_LINE_PX),
                alpha=0.8,
                zorder=2,
            )
            drawn += 1
        return drawn

    @staticmethod
    def _draw_stars(
        axes,
        stars: Sequence[CatalogStar],
        star_altaz: dict[str, tuple[float, float]],
    ) -> int:
        x_values: list[float] = []
        y_values: list[float] = []
        sizes: list[float] = []
        for star in sort_stars_dim_to_bright(stars):
            altitude, azimuth = star_altaz[star.star_id]
            if altitude < 0:
                continue
            x, y = project_altaz(altitude, azimuth)
            x_values.append(x)
            y_values.append(y)
            sizes.append(
                marker_area(
                    max(
                        _STAR_FLOOR_PX,
                        _STAR_BRIGHT_PX - _STAR_MAGNITUDE_FALLOFF_PX * star.magnitude,
                    )
                )
            )
        if x_values:
            axes.scatter(
                x_values,
                y_values,
                s=sizes,
                c="#f5f1df",
                edgecolors="none",
                zorder=3,
            )
        return len(x_values)

    @staticmethod
    def _draw_moon(axes, moon: SkyChartObject, labels: "_LabelPlacer") -> None:
        if not moon.drawn or moon.altaz.altitude_deg < 0:
            return
        x, y = project_altaz(moon.altaz.altitude_deg, moon.altaz.azimuth_deg)
        illumination = moon.illumination_fraction or 0.0
        radius = 0.022
        axes.add_patch(
            Circle((x, y), radius, facecolor="#252830", edgecolor="none", zorder=4)
        )
        axes.add_patch(
            Wedge(
                (x, y),
                radius,
                theta1=90,
                theta2=90 + 360 * illumination,
                facecolor="#f2ead2",
                edgecolor="none",
                zorder=4.1,
            )
        )
        axes.add_patch(
            Circle(
                (x, y),
                radius,
                fill=False,
                edgecolor="#f1ead4",
                linewidth=pt_from_canvas_px(_MOON_OUTLINE_PX),
                zorder=4.2,
            )
        )
        labels.add(
            x,
            y,
            moon.label,
            color="#e8e1ca",
            fontsize=pt_from_canvas_px(_OBJECT_LABEL_FONT_PX),
            zorder=4,
        )

    @staticmethod
    def _draw_planets(
        axes,
        planets: Sequence[SkyChartObject],
        planet_records,
        labels: "_LabelPlacer",
    ) -> None:
        for planet, (_body_name, _label, color, _coordinate) in zip(planets, planet_records, strict=True):
            if not planet.drawn:
                continue
            x, y = project_altaz(planet.altaz.altitude_deg, planet.altaz.azimuth_deg)
            axes.scatter(
                [x],
                [y],
                s=marker_area(_PLANET_MARKER_PX),
                c=color,
                edgecolors="#ffffff",
                linewidths=pt_from_canvas_px(_PLANET_EDGE_PX),
                zorder=5,
            )
            labels.add(
                x,
                y,
                planet.label,
                color=color,
                fontsize=pt_from_canvas_px(_OBJECT_LABEL_FONT_PX),
                zorder=5,
            )

    @staticmethod
    def _draw_target(axes, target: SkyChartObject | None, labels: "_LabelPlacer") -> None:
        if target is None or not target.drawn:
            return
        x, y = project_altaz(target.altaz.altitude_deg, target.altaz.azimuth_deg)
        axes.scatter(
            [x],
            [y],
            s=marker_area(_TARGET_RING_PX),
            facecolors="none",
            edgecolors="#ffd43b",
            linewidths=pt_from_canvas_px(_TARGET_RING_LINE_PX),
            zorder=6,
        )
        axes.scatter(
            [x],
            [y],
            s=marker_area(_TARGET_CROSS_PX),
            marker="+",
            c="#ffd43b",
            linewidths=pt_from_canvas_px(_TARGET_CROSS_LINE_PX),
            zorder=6,
        )
        labels.add(
            x,
            y,
            target.label,
            color="#ffd43b",
            fontsize=pt_from_canvas_px(_TARGET_LABEL_FONT_PX),
            zorder=6,
        )

    @staticmethod
    def _draw_footer(figure: Figure, request: SkyChartRequest, context: _RenderContext, selection: CatalogSelection) -> None:
        # Two lines, not one: at the size the footer needs to stay legible on screen
        # the single 144-character line would overflow the canvas. The observer's
        # place name is free text, so the block is measured and fitted rather than
        # assumed to fit.
        local = request.timestamp_local.isoformat()
        utc = context.timestamp_utc.isoformat().replace("+00:00", "Z")
        lines, fontsize_px = fit_footer_lines(
            (
                f"{request.observer.location_name} | {request.observer.timezone} | {local} | UTC {utc}",
                f"catalog {selection.mode_used}/{selection.status} | AltAz pressure=0 hPa | builtin ephemeris",
            ),
            _FOOTER_FONT_PX,
        )
        fontsize = pt_from_canvas_px(fontsize_px)
        for y, text in zip((0.0495, 0.0260), lines, strict=True):
            figure.text(0.5, y, text, color="#93a2aa", fontsize=fontsize, ha="center", va="center")

    @staticmethod
    def _save_rgb_png(figure: Figure) -> bytes:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                # A place name may still carry a glyph no installed font covers.
                # The chart renders correctly apart from that one character, so
                # keep it out of stderr instead of failing the request.
                message=r"Glyph .* missing from font\(s\) .*\.",
                category=UserWarning,
            )
            canvas = figure.canvas
            canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        height, width, channels = rgba.shape
        if channels != 4:
            raise ValueError("Agg canvas did not provide RGBA pixels")
        rgb_bytes = np.ascontiguousarray(rgba[:, :, :3]).tobytes()
        return _encode_rgb_png(rgb_bytes, width=width, height=height)

    @staticmethod
    def _target_input(request: SkyChartRequest) -> str:
        if request.target.mode == "name":
            assert request.target.name is not None
            return request.target.name
        assert request.target.ra_deg is not None and request.target.dec_deg is not None
        return f"{request.target.ra_deg:.6f}, {request.target.dec_deg:.6f}"


class SkyChartService:
    """Select local data, contain resolver failures, and render one chart."""

    def __init__(
        self,
        *,
        full_catalog_cache: _FullCatalogCache | None = None,
        target_resolver: SkyChartTargetResolver | None = None,
        target_cache_dir: Path = Path("cache/targets"),
        target_backend: TargetBackend | None = None,
        bundled_catalog: BundledCatalog | None = None,
        utc_clock: Callable[[], datetime] = utc_now,
        renderer: SkyChartRenderer | None = None,
    ) -> None:
        self._bundled_catalog = bundled_catalog or load_bundled_catalog()
        self._full_catalog_cache = full_catalog_cache or _EmptyFullCatalogCache()
        if target_resolver is None:
            backend = target_backend or _LazySimbadBackend()
            target_resolver = SkyChartTargetResolver(
                lambda name: resolve_target(
                    name,
                    backend=backend,
                    cache_dir=target_cache_dir,
                )
            )
        self._target_resolver = target_resolver
        self._renderer = renderer or SkyChartRenderer(utc_clock=utc_clock)

    def render(self, request: SkyChartRequest) -> RenderedSkyChart:
        selection = select_catalog(
            request.catalog_mode,
            self._bundled_catalog,
            self._full_catalog_cache,
        )
        warning: str | None = None
        try:
            resolved = self._target_resolver.resolve(request.target)
        except (InvalidTargetNameError, TargetNotFoundError, UnsupportedSolarSystemBodyError):
            resolved = None
            warning = "target_unresolved"
        except TargetServiceError:
            resolved = None
            warning = "target_resolution_unavailable"
        else:
            if resolved is None:
                warning = "target_unresolved"

        chart = self._renderer.render(request, selection, resolved)
        if warning is None:
            return chart
        metadata = chart.metadata.model_copy(
            update={"warnings": [*chart.metadata.warnings, warning]}
        )
        return RenderedSkyChart(
            png_bytes=chart.png_bytes,
            metadata=metadata,
            catalog_mode_used=chart.catalog_mode_used,
            catalog_status=chart.catalog_status,
        )


class RenderStore:
    """Thread-safe TTL store containing only the two export byte payloads."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 15 * 60,
        max_records: int = 20,
        max_bytes: int = 50 * 1024 * 1024,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0 or max_records <= 0 or max_bytes <= 0:
            raise ValueError("render store limits must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_records = int(max_records)
        self.max_bytes = int(max_bytes)
        self._monotonic_clock = monotonic_clock or __import__("time").monotonic
        self._records: dict[str, _StoreRecord] = {}
        self._total_bytes = 0
        self._insertion_order = 0
        self._lock = threading.Lock()

    def put(self, chart: RenderedSkyChart) -> str:
        with self._lock:
            now = self._monotonic_clock()
            self._purge_expired(now)
            if chart.metadata.render.png_sha256 != sha256(chart.png_bytes).hexdigest():
                raise ValueError("render metadata does not match PNG bytes")
            _validate_rgb_png(chart.png_bytes)
            render_id = self._new_render_id()
            metadata = chart.metadata.model_copy(update={"render_id": render_id})
            metadata_json = _serialize_metadata(metadata)
            self._put_record(render_id, now, chart.png_bytes, metadata_json)
            return render_id

    def get(self, render_id: str) -> RenderedSkyChart | None:
        with self._lock:
            now = self._monotonic_clock()
            self._purge_expired(now)
            if not isinstance(render_id, str) or not _RENDER_ID_RE.fullmatch(render_id):
                return None
            record = self._records.get(render_id)
            if record is None:
                return None
            metadata = SkyChartExportMetadata.model_validate_json(
                record.metadata_json_bytes
            )
            return RenderedSkyChart._from_store(
                png_bytes=record.png_bytes,
                metadata=metadata,
                catalog_mode_used=metadata.request.catalog_mode_used,
                catalog_status=metadata.catalog.status,
                metadata_json_bytes=record.metadata_json_bytes,
            )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._total_bytes = 0

    def _put_record(
        self,
        render_id: str,
        now: float,
        png_bytes: bytes,
        metadata_json_bytes: bytes,
    ) -> None:
        byte_size = len(png_bytes) + len(metadata_json_bytes)
        if byte_size > self.max_bytes:
            raise ValueError("render exceeds the store byte capacity")
        while self._records and (
            len(self._records) >= self.max_records
            or self._total_bytes + byte_size > self.max_bytes
        ):
            eviction_id = min(
                self._records,
                key=lambda candidate: (
                    self._records[candidate].expires_at,
                    self._records[candidate].insertion_order,
                ),
            )
            self._remove(eviction_id)
        record = _StoreRecord(
            expires_at=now + self.ttl_seconds,
            insertion_order=self._insertion_order,
            png_bytes=png_bytes,
            metadata_json_bytes=metadata_json_bytes,
        )
        self._insertion_order += 1
        self._records[render_id] = record
        self._total_bytes += record.byte_size

    def _purge_expired(self, now: float) -> None:
        for render_id in [
            candidate
            for candidate, record in self._records.items()
            if record.expires_at <= now
        ]:
            self._remove(render_id)

    def _remove(self, render_id: str) -> None:
        record = self._records.pop(render_id)
        self._total_bytes -= record.byte_size

    def _new_render_id(self) -> str:
        while True:
            render_id = secrets.token_urlsafe(24)
            if _RENDER_ID_RE.fullmatch(render_id) and render_id not in self._records:
                return render_id


def _dependency_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "system"


def _serialize_metadata(metadata: SkyChartExportMetadata) -> bytes:
    return metadata.model_dump_json(
        exclude_none=False,
        by_alias=True,
    ).encode("utf-8")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _encode_rgb_png(rgb_bytes: bytes, *, width: int, height: int) -> bytes:
    row_size = width * 3
    if len(rgb_bytes) != row_size * height:
        raise ValueError("RGB pixel buffer has an unexpected size")

    scanlines = bytearray((row_size + 1) * height)
    for row in range(height):
        source_start = row * row_size
        target_start = row * (row_size + 1)
        scanlines[target_start + 1 : target_start + row_size + 1] = rgb_bytes[
            source_start : source_start + row_size
        ]

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=6)),
            _png_chunk(b"IEND", b""),
        )
    )


def _validate_rgb_png(png_bytes: bytes) -> None:
    error = ValueError(
        f"render must be a valid {CANVAS_WIDTH_PX}x{CANVAS_HEIGHT_PX} RGB PNG"
    )
    signature = b"\x89PNG\r\n\x1a\n"
    if not isinstance(png_bytes, bytes) or not png_bytes.startswith(signature):
        raise error

    offset = len(signature)
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(png_bytes):
        if len(png_bytes) - offset < 12:
            raise error
        length = struct.unpack(">I", png_bytes[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(png_bytes):
            raise error
        chunk_type = png_bytes[offset + 4 : offset + 8]
        data = png_bytes[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", png_bytes[offset + 8 + length : chunk_end])[0]
        calculated_crc = zlib.crc32(data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if stored_crc != calculated_crc:
            raise error
        chunks.append((chunk_type, data))
        offset = chunk_end
        if chunk_type == b"IEND":
            break

    if offset != len(png_bytes) or [kind for kind, _data in chunks] != [
        b"IHDR",
        b"IDAT",
        b"IEND",
    ]:
        raise error

    ihdr = chunks[0][1]
    if len(ihdr) != 13 or chunks[2][1]:
        raise error
    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if (
        width,
        height,
        bit_depth,
        color_type,
        compression,
        filter_method,
        interlace,
    ) != (CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX, 8, 2, 0, 0, 0):
        raise error

    row_size = width * 3
    expected_size = (row_size + 1) * height
    decompressor = zlib.decompressobj()
    try:
        scanlines = decompressor.decompress(chunks[1][1], expected_size + 1)
        if len(scanlines) > expected_size or decompressor.unconsumed_tail:
            raise error
        scanlines += decompressor.flush(expected_size - len(scanlines) + 1)
    except zlib.error as exc:
        raise error from exc
    if (
        len(scanlines) != expected_size
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or any(scanlines[row * (row_size + 1)] != 0 for row in range(height))
    ):
        raise error
