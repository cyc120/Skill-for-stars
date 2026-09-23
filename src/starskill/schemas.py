"""Structured input models for observation tasks."""

from datetime import datetime, timedelta, timezone
import re
from typing import Annotated, Literal, Protocol
import unicodedata
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_SKY_CHART_COORDINATE_JSON_PATTERN = re.compile(
    r'("(?:longitude|latitude|ra_deg|dec_deg|altitude_deg|azimuth_deg)"\s*:\s*)'
    r'(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s*[,}])'
)


class Observer(InputModel):
    location_name: str
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    timezone: str

    @field_validator("timezone")
    @classmethod
    def timezone_must_be_iana_name(cls, value: str) -> str:
        value = value.strip()
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value


class TimeRange(InputModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "TimeRange":
        if self.end <= self.start:
            raise ValueError("end must be later than start")
        return self


def require_non_empty_utc_range(time_range: TimeRange, observer: Observer) -> None:
    """Reject a local window that is empty once resolved to real instants.

    ``TimeRange.end_must_follow_start`` compares the wall-clock fields, which is
    not the order the sampler uses. On a spring-forward night a local window such
    as 02:30-03:00 names an hour that never happens and runs backwards in UTC, so
    the sample grid would come out empty and every downstream command would fail
    with an unhelpful error. Resolving both ends through the observer's zone and
    comparing the instants catches that here, where it is still input validation.
    """
    zone = ZoneInfo(observer.timezone)
    start = (
        time_range.start
        if time_range.start.tzinfo is not None
        else time_range.start.replace(tzinfo=zone)
    )
    end = (
        time_range.end
        if time_range.end.tzinfo is not None
        else time_range.end.replace(tzinfo=zone)
    )
    if end.astimezone(timezone.utc) <= start.astimezone(timezone.utc):
        raise ValueError(
            "time_range must span a non-empty interval in real time: "
            f"{time_range.start.isoformat()} to {time_range.end.isoformat()} is empty "
            f"or runs backwards in {observer.timezone}; a daylight-saving transition "
            "may skip the requested local hour"
        )


class OutputOptions(InputModel):
    language: str = "zh-CN"
    level: str = "classroom"
    formats: list[str] = Field(default_factory=lambda: ["json", "csv", "png", "md"])


class ObservationTask(InputModel):
    task_type: Literal["observation_plan"] = "observation_plan"
    target: "str | TargetRef"
    observer: Observer
    time_range: TimeRange
    interval_minutes: int = Field(default=10, ge=1, le=120)
    output: OutputOptions = Field(default_factory=OutputOptions)

    @model_validator(mode="after")
    def normalize_legacy_target(self) -> "ObservationTask":
        require_non_empty_utc_range(self.time_range, self.observer)
        if isinstance(self.target, str):
            name = self.target.strip()
            if not name:
                raise ValueError("target must not be blank")
            self.target = SimbadTargetRef(kind="simbad", name=name)
        return self


class TargetSource(InputModel):
    database: Literal["SIMBAD"]
    service_url: str
    accessed_at: datetime
    from_cache: bool


class ResolvedTarget(InputModel):
    input_name: str
    query_name: str
    canonical_name: str
    ra_deg: float = Field(ge=0, lt=360)
    dec_deg: float = Field(ge=-90, le=90)
    object_type: str
    aliases: list[str]
    coordinate_frame: Literal["ICRS"] = "ICRS"
    source: TargetSource


class EphemerisSample(InputModel):
    timestamp_local: datetime
    timestamp_utc: datetime
    target_altitude_deg: float = Field(ge=-90, le=90)
    target_azimuth_deg: float = Field(ge=0, lt=360)
    sun_altitude_deg: float = Field(ge=-90, le=90)
    moon_altitude_deg: float = Field(ge=-90, le=90)
    moon_separation_deg: float = Field(ge=0, le=180)

    @field_validator("timestamp_local", "timestamp_utc")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value

    @field_validator("timestamp_utc")
    @classmethod
    def utc_timestamp_must_use_zero_offset(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must use UTC offset +00:00")
        return value


class EphemerisSettings(InputModel):
    calculated_at: datetime
    astropy_version: str
    time_scale: Literal["UTC"] = "UTC"
    horizontal_frame: Literal["AltAz"] = "AltAz"
    atmospheric_refraction: bool = False
    iers_auto_download: bool = False


class EphemerisResult(InputModel):
    target: "ResolvedAstronomicalTarget"
    observer: Observer
    interval_minutes: int = Field(ge=1)
    settings: EphemerisSettings
    samples: list[EphemerisSample] = Field(min_length=1)


VisibilityRejectionReason = Literal[
    "target_below_minimum_altitude",
    "sun_above_maximum_altitude",
]


class VisibilityCriteria(InputModel):
    min_target_altitude_deg: float = Field(default=30.0, ge=-90, le=90)
    max_sun_altitude_deg: float = Field(default=-12.0, ge=-90, le=90)


class VisibilitySample(EphemerisSample):
    moon_illumination_fraction: float = Field(ge=0, le=1)
    is_observable: bool
    rejection_reasons: list[VisibilityRejectionReason]


class ObservationWindow(InputModel):
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime
    sample_count: int = Field(ge=1)
    peak_target_altitude_deg: float = Field(ge=-90, le=90)


class ObservationPlanResult(InputModel):
    target: "ResolvedAstronomicalTarget"
    observer: Observer
    interval_minutes: int = Field(ge=1)
    source_ephemeris_settings: EphemerisSettings
    criteria: VisibilityCriteria
    samples: list[VisibilitySample] = Field(min_length=1)
    windows: list[ObservationWindow]


class PipelineIssue(InputModel):
    stage: str
    code: str
    message: str


class ArtifactRecord(InputModel):
    path: str
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PipelineManifest(InputModel):
    run_id: str
    status: Literal["success", "degraded", "failed"]
    started_at: datetime
    completed_at: datetime
    input_task: ObservationTask
    cache_hit: bool
    target_source: "TargetSource | AstronomicalTargetSource | None"
    dependencies: dict[str, str]
    artifacts: list[ArtifactRecord]
    issues: list[PipelineIssue]


class PipelineOutcome(InputModel):
    status: Literal["success", "degraded", "failed"]
    output_dir: str
    manifest: PipelineManifest


class SolarSystemRelationshipTask(InputModel):
    task_type: Literal["solar_system_relationship"] = "solar_system_relationship"
    targets: list[Literal["moon", "jupiter"]]
    observer: Observer
    time_range: TimeRange
    interval_minutes: int = Field(default=20, ge=1, le=120)

    @model_validator(mode="after")
    def require_moon_and_jupiter(self) -> "SolarSystemRelationshipTask":
        require_non_empty_utc_range(self.time_range, self.observer)
        if self.targets != ["moon", "jupiter"]:
            raise ValueError("targets must be exactly ['moon', 'jupiter']")
        return self


class SolarSystemRelationshipSettings(InputModel):
    calculated_at: datetime
    astropy_version: str
    time_scale: Literal["UTC"] = "UTC"
    horizontal_frame: Literal["AltAz"] = "AltAz"
    solar_system_ephemeris: Literal["builtin"] = "builtin"
    atmospheric_refraction: bool = False
    iers_auto_download: bool = False


class SolarSystemRelationshipSample(InputModel):
    timestamp_local: datetime
    timestamp_utc: datetime
    moon_altitude_deg: float = Field(ge=-90, le=90)
    moon_azimuth_deg: float = Field(ge=0, lt=360)
    jupiter_altitude_deg: float = Field(ge=-90, le=90)
    jupiter_azimuth_deg: float = Field(ge=0, lt=360)
    angular_separation_deg: float = Field(ge=0, le=180)


class SolarSystemRelationshipResult(InputModel):
    task: SolarSystemRelationshipTask
    settings: SolarSystemRelationshipSettings
    samples: list[SolarSystemRelationshipSample] = Field(min_length=1)


class SolarSystemTargetRef(InputModel):
    kind: Literal["solar_system"]
    body: str

    @field_validator("body")
    @classmethod
    def normalize_body(cls, value: str) -> str:
        body = " ".join(value.split()).casefold()
        if not body or len(body) > 64 or not re.fullmatch(r"[a-z][a-z0-9_ -]*", body):
            raise ValueError("solar-system body must be a safe non-empty name")
        return body


class SimbadTargetRef(InputModel):
    kind: Literal["simbad"]
    name: str


class CoordinateTargetRef(InputModel):
    kind: Literal["coordinates"]
    label: str = Field(min_length=1, max_length=120)
    ra_deg: float = Field(ge=0, lt=360, allow_inf_nan=False)
    dec_deg: float = Field(ge=-90, le=90, allow_inf_nan=False)


TargetRef = Annotated[
    SolarSystemTargetRef | SimbadTargetRef | CoordinateTargetRef,
    Field(discriminator="kind"),
]


ImageProviderId = Literal["sdss_dr18", "mast", "esa_sky", "panstarrs"]
ImageFormat = Literal["jpeg", "png", "fits"]
ImageProviderMode = Literal[
    "auto_trusted",
    "sdss_dr18",
    "mast",
    "esa_sky",
    "panstarrs",
]


class ImageContractModel(InputModel):
    """Immutable boundary models for generic image discovery and retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_https_url(value: str) -> str:
    normalized = value.strip()
    parsed = urlsplit(normalized)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("URL must be an absolute HTTPS URL without credentials")
    return normalized


def _require_hostname(value: str) -> str:
    normalized = value.strip().casefold()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", normalized):
        raise ValueError("host must be a DNS hostname")
    return normalized


class AstronomyImageSearchRequest(ImageContractModel):
    target: TargetRef
    observed_at: datetime | None = None
    field_of_view_arcmin: float = Field(default=12, gt=0, le=120)
    bands: list[str] = Field(default_factory=list, max_length=8)
    max_width: int = Field(default=2048, ge=64, le=4096)
    max_height: int = Field(default=2048, ge=64, le=4096)
    allowed_formats: list[ImageFormat] = Field(
        default_factory=lambda: ["jpeg", "png", "fits"]
    )
    timeout_seconds: int = Field(default=30, ge=1, le=120)
    max_bytes: int = Field(default=20_000_000, ge=1, le=50_000_000)
    provider_mode: ImageProviderMode = "auto_trusted"

    @model_validator(mode="after")
    def require_dynamic_target_time(self) -> "AstronomyImageSearchRequest":
        if isinstance(self.target, SolarSystemTargetRef) and self.observed_at is None:
            raise ValueError("solar-system image requests require observed_at")
        if self.observed_at is not None and (
            self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None
        ):
            raise ValueError("observed_at must include a timezone offset")
        return self


class ResolvedImageTarget(ImageContractModel):
    label: str = Field(min_length=1, max_length=120)
    ra_deg: float = Field(ge=0, lt=360, allow_inf_nan=False)
    dec_deg: float = Field(ge=-90, le=90, allow_inf_nan=False)
    coordinate_frame: Literal["ICRS"] = "ICRS"
    source: "AstronomicalTargetSource"
    observed_at: datetime | None = None

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_include_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at must include a timezone offset")
        return value


class ImageCandidate(ImageContractModel):
    candidate_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    provider_id: ImageProviderId
    source_url: str
    download_url: str
    band: str | None = Field(default=None, min_length=1, max_length=64)
    format: ImageFormat
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    query_parameters: dict[str, str | int | float] = Field(default_factory=dict)
    license_url: str | None = None

    @field_validator("source_url", "download_url")
    @classmethod
    def urls_must_be_https(cls, value: str) -> str:
        return _require_https_url(value)

    @field_validator("license_url")
    @classmethod
    def license_url_must_be_https(cls, value: str | None) -> str | None:
        return _require_https_url(value) if value is not None else None


class ImageProviderDescriptor(ImageContractModel):
    provider_id: ImageProviderId
    organization: str = Field(min_length=1, max_length=160)
    allowed_hosts: tuple[str, ...] = Field(min_length=1)
    allowed_redirect_hosts: tuple[str, ...] = Field(min_length=1)
    endpoint_roots: tuple[str, ...] = Field(min_length=1)
    formats: tuple[ImageFormat, ...] = Field(min_length=1)
    max_bytes: int = Field(ge=1, le=50_000_000)
    license_url: str

    @field_validator("allowed_hosts", "allowed_redirect_hosts")
    @classmethod
    def hosts_must_be_dns_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_require_hostname(host) for host in value)

    @field_validator("endpoint_roots")
    @classmethod
    def endpoint_roots_must_be_https(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_require_https_url(root) for root in value)

    @field_validator("license_url")
    @classmethod
    def license_url_must_be_https(cls, value: str) -> str:
        return _require_https_url(value)


class ImageTrustDecision(ImageContractModel):
    allowed: bool
    reason_code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+$")


class ImageRank(ImageContractModel):
    candidate_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    relevance: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=1, max_length=500)

    @classmethod
    def zero(cls, candidate_id: str) -> "ImageRank":
        return cls(
            candidate_id=candidate_id,
            confidence=0.0,
            relevance=0.0,
            reason="no_model_rank",
        )


class ModelRanker(Protocol):
    def rank(
        self,
        request: AstronomyImageSearchRequest,
        candidates: list[ImageCandidate],
    ) -> list[ImageRank]: ...


class ImageSearchResult(ImageContractModel):
    request: AstronomyImageSearchRequest
    target: ResolvedImageTarget
    candidates: list[ImageCandidate] = Field(default_factory=list)
    ranks: list[ImageRank] = Field(default_factory=list)
    decisions: dict[str, ImageTrustDecision] = Field(default_factory=dict)
    selected_candidate: ImageCandidate | None = None

    @model_validator(mode="after")
    def ranks_must_cover_each_candidate_once(self) -> "ImageSearchResult":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")

        rank_ids = [rank.candidate_id for rank in self.ranks]
        if len(rank_ids) != len(set(rank_ids)):
            raise ValueError("model ranks must contain one unique rank per candidate")
        if self.ranks and set(rank_ids) != set(candidate_ids):
            raise ValueError("model ranks must contain one rank for every input candidate")

        if (
            self.selected_candidate is not None
            and self.selected_candidate.candidate_id not in candidate_ids
        ):
            raise ValueError("selected candidate must come from discovered candidates")
        return self


class AstronomicalRelationshipTask(InputModel):
    task_type: Literal["astronomical_relationship"] = "astronomical_relationship"
    primary: TargetRef
    secondary: TargetRef
    observer: Observer
    time_range: TimeRange
    interval_minutes: int = Field(default=20, ge=1, le=120)

    @model_validator(mode="after")
    def require_non_empty_time_range(self) -> "AstronomicalRelationshipTask":
        require_non_empty_utc_range(self.time_range, self.observer)
        return self


class AstronomicalTargetSource(InputModel):
    provider: str
    from_cache: bool
    accessed_at: datetime


class ResolvedAstronomicalTarget(InputModel):
    label: str
    kind: Literal["solar_system", "simbad", "coordinates"]
    motion: Literal["dynamic", "fixed_icrs"]
    ra_deg: float | None = Field(default=None, ge=0, lt=360, allow_inf_nan=False)
    dec_deg: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    source: AstronomicalTargetSource
    catalog_target: ResolvedTarget | None = None

    @property
    def canonical_name(self) -> str:
        """Retain the display-name attribute used by observation renderers."""
        return self.label

    @model_validator(mode="after")
    def motion_must_match_kind_and_coordinates(self) -> "ResolvedAstronomicalTarget":
        if self.kind == "solar_system":
            if self.motion != "dynamic":
                raise ValueError("solar_system targets must use dynamic motion")
            if self.ra_deg is not None or self.dec_deg is not None:
                raise ValueError("dynamic solar_system targets must not have fixed ICRS coordinates")
            return self

        if self.motion != "fixed_icrs":
            raise ValueError(f"{self.kind} targets must use fixed_icrs motion")
        if self.ra_deg is None or self.dec_deg is None:
            raise ValueError(f"fixed_icrs {self.kind} targets require ra_deg and dec_deg")
        return self


class AstronomicalRelationshipSettings(InputModel):
    schema_version: Literal["2.0"] = "2.0"
    calculated_at: datetime
    astropy_version: str
    time_scale: Literal["UTC"] = "UTC"
    horizontal_frame: Literal["AltAz"] = "AltAz"
    solar_system_ephemeris: Literal["builtin"] = "builtin"
    atmospheric_refraction: bool = False
    iers_auto_download: bool = False


class AstronomicalRelationshipSample(InputModel):
    timestamp_local: datetime
    timestamp_utc: datetime
    primary_altitude_deg: float = Field(ge=-90, le=90)
    primary_azimuth_deg: float = Field(ge=0, lt=360)
    primary_is_above_horizon: bool
    secondary_altitude_deg: float = Field(ge=-90, le=90)
    secondary_azimuth_deg: float = Field(ge=0, lt=360)
    secondary_is_above_horizon: bool
    angular_separation_deg: float = Field(ge=0, le=180)

    @model_validator(mode="after")
    def horizon_flags_must_match_altitudes(self) -> "AstronomicalRelationshipSample":
        if self.primary_is_above_horizon != (self.primary_altitude_deg >= 0):
            raise ValueError(
                "primary_is_above_horizon must equal primary_altitude_deg >= 0"
            )
        if self.secondary_is_above_horizon != (self.secondary_altitude_deg >= 0):
            raise ValueError(
                "secondary_is_above_horizon must equal secondary_altitude_deg >= 0"
            )
        return self


class AstronomicalRelationshipResult(InputModel):
    task: AstronomicalRelationshipTask
    primary: ResolvedAstronomicalTarget
    secondary: ResolvedAstronomicalTarget
    settings: AstronomicalRelationshipSettings
    samples: list[AstronomicalRelationshipSample] = Field(min_length=1)


class SDSSImageRequest(InputModel):
    target_name: Literal["M51"] = "M51"
    data_release: Literal["DR18"] = "DR18"
    ra_deg: float = Field(default=202.4696, ge=0, lt=360)
    dec_deg: float = Field(default=47.1952, ge=-90, le=90)
    scale_arcsec_per_pixel: float = Field(default=0.396, gt=0, le=10)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    timeout_seconds: int = Field(default=30, ge=1, le=120)
    max_bytes: int = Field(default=5_000_000, ge=1, le=20_000_000)


class SDSSImageSource(InputModel):
    database: Literal["SDSS SkyServer"] = "SDSS SkyServer"
    data_release: Literal["DR18"] = "DR18"
    endpoint: str
    source_url: str
    accessed_at: datetime
    from_cache: bool
    authentication: Literal["none"] = "none"
    query_parameters: dict[str, str | int | float]
    expected_count: int = 1
    retrieved_count: int = 1
    local_filters: list[str] = Field(default_factory=list)
    content_type: str
    bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pixel_scale_arcsec: float = Field(gt=0)
    wavebands: list[str]
    license_notice: str


class PublicImageResult(InputModel):
    request: SDSSImageRequest
    source: SDSSImageSource
    source_path: str
    display_path: str
    processing_steps: list[str]


ExternalAvailability = Literal["fresh", "cached", "unavailable", "stale"]


class ExternalSource(InputModel):
    provider: str
    source_url: str | None = None
    accessed_at: datetime
    from_cache: bool
    availability: ExternalAvailability
    issue_code: str | None = None


class ObservingConditionsRequest(InputModel):
    observer: Observer
    time_range: TimeRange


class WeatherSample(InputModel):
    timestamp_local: datetime
    cloud_cover_percent: float | None = Field(default=None, ge=0, le=100)
    precipitation_mm: float | None = Field(default=None, ge=0)
    wind_speed_kmh: float | None = Field(default=None, ge=0)
    visibility_m: float | None = Field(default=None, ge=0)


class WeatherForecast(InputModel):
    samples: list[WeatherSample]
    source: ExternalSource


class LightPollutionResult(InputModel):
    radiance: float | None = Field(default=None, ge=0)
    unit: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    sample_period: str | None = None
    spatial_resolution: str | None = None
    interpolation: str | None = None
    source: ExternalSource


class NasaFeature(InputModel):
    date: str | None = None
    title: str | None = None
    media_type: str | None = None
    media_url: str | None = None
    explanation: str | None = None
    copyright: str | None = None
    source: ExternalSource


class SkyChartObserver(InputModel):
    """Observer input dedicated to the local, deterministic sky chart."""

    location_name: str = Field(default="北京", min_length=1, max_length=80)
    longitude: float = Field(default=116.4074, ge=-180, le=180, allow_inf_nan=False)
    latitude: float = Field(default=39.9042, ge=-90, le=90, allow_inf_nan=False)
    timezone: str = "Asia/Shanghai"

    @field_validator("location_name")
    @classmethod
    def normalize_location_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(character) < 32 for character in value):
            raise ValueError("location_name must contain 1..80 visible characters")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return Observer(location_name="x", longitude=0, latitude=0, timezone=value).timezone

    @field_serializer("longitude", "latitude", when_used="json")
    def serialize_six_decimal_coordinate(self, value: float) -> float:
        return round(value, 6)


class SkyChartTarget(InputModel):
    """Mutually exclusive target-name or ICRS-coordinate input."""

    mode: Literal["name", "coordinates"] = "name"
    name: str | None = "M42"
    ra_deg: float | None = Field(default=None, allow_inf_nan=False)
    dec_deg: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def remove_name_default_for_coordinate_input(cls, value: object) -> object:
        if isinstance(value, dict) and value.get("mode") == "coordinates" and "name" not in value:
            return {**value, "name": None}
        return value

    @model_validator(mode="after")
    def enforce_target_mode(self) -> "SkyChartTarget":
        if self.mode == "name":
            if self.ra_deg is not None or self.dec_deg is not None or not self.name:
                raise ValueError("name target requires only a visible 1..120 character name")
            if any(
                unicodedata.category(character) in {"Cc", "Cf"}
                for character in self.name
            ):
                raise ValueError("name target requires only a safe visible 1..120 character name")
            name = self.name.strip()
            if (
                not name
                or len(name) > 120
                or any(
                    character in name
                    for character in ":/?#&%\\\"';|<>`$(){}[]*!~"
                )
            ):
                raise ValueError("name target requires only a safe visible 1..120 character name")
            self.name = name
        elif (
            self.name is not None
            or self.ra_deg is None
            or self.dec_deg is None
            or not 0 <= self.ra_deg < 360
            or not -90 <= self.dec_deg <= 90
        ):
            raise ValueError("coordinates target requires only ra_deg and dec_deg")
        return self


class SkyChartRequest(InputModel):
    """All and only client-controlled sky-chart inputs."""

    observer: SkyChartObserver = Field(default_factory=SkyChartObserver)
    timestamp_local: datetime
    target: SkyChartTarget = Field(default_factory=SkyChartTarget)
    catalog_mode: Literal["auto", "bundled", "full"] = "auto"

    @field_validator("timestamp_local")
    @classmethod
    def require_offset(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp_local must include a timezone offset")
        return value

    @model_validator(mode="after")
    def require_matching_zone_offset(self) -> "SkyChartRequest":
        zone_offset = self.timestamp_local.astimezone(
            ZoneInfo(self.observer.timezone)
        ).utcoffset()
        if self.timestamp_local.utcoffset() != zone_offset:
            raise ValueError("timestamp_local offset must match observer timezone")
        return self


_SKY_CHART_RENDER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SKY_CHART_LAYER_ORDER = [
    "background",
    "horizon_grid",
    "constellations",
    "stars",
    "moon",
    "planets",
    "target",
    "footer",
]


class SkyChartRenderResponse(InputModel):
    render_id: str = Field(pattern=_SKY_CHART_RENDER_ID_PATTERN)
    png_url: str
    json_url: str
    catalog_mode_used: Literal["bundled", "full"]
    catalog_status: Literal["available", "degraded"]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_same_origin_render_urls(self) -> "SkyChartRenderResponse":
        base = f"/v1/sky-chart/renders/{self.render_id}"
        if self.png_url != f"{base}.png" or self.json_url != f"{base}.json":
            raise ValueError("render URLs must be same-origin URLs for render_id")
        return self


class SkyChartIcrsCoordinates(InputModel):
    ra_deg: float = Field(ge=0, lt=360, allow_inf_nan=False)
    dec_deg: float = Field(ge=-90, le=90, allow_inf_nan=False)

    @field_serializer("ra_deg", "dec_deg", when_used="json")
    def serialize_six_decimal_coordinate(self, value: float) -> float:
        return round(value, 6)


class SkyChartAltAzCoordinates(InputModel):
    altitude_deg: float = Field(ge=-90, le=90, allow_inf_nan=False)
    azimuth_deg: float = Field(ge=0, lt=360, allow_inf_nan=False)

    @field_serializer("altitude_deg", "azimuth_deg", when_used="json")
    def serialize_six_decimal_coordinate(self, value: float) -> float:
        return round(value, 6)


class SkyChartObject(InputModel):
    label: str = Field(min_length=1, max_length=120)
    icrs: SkyChartIcrsCoordinates | None
    altaz: SkyChartAltAzCoordinates
    visible: bool
    drawn: bool
    illumination_fraction: float | None = Field(default=None, ge=0, le=1)


class SkyChartExportTarget(InputModel):
    mode: Literal["name", "coordinates"]
    input: str = Field(min_length=1, max_length=120)
    resolved: SkyChartObject | None


class SkyChartExportRequest(InputModel):
    observer: SkyChartObserver
    timestamp_local: datetime
    timestamp_utc: datetime
    target: SkyChartExportTarget
    catalog_mode_requested: Literal["auto", "bundled", "full"]
    catalog_mode_used: Literal["bundled", "full"]

    @field_validator("timestamp_local")
    @classmethod
    def timestamp_local_must_include_offset(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp_local must include a timezone offset")
        return value

    @field_validator("timestamp_utc")
    @classmethod
    def timestamp_utc_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must use UTC offset")
        return value

    @field_serializer("timestamp_utc", when_used="json")
    def serialize_timestamp_utc(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def timestamps_must_describe_one_instant(self) -> "SkyChartExportRequest":
        if self.timestamp_local.astimezone(timezone.utc) != self.timestamp_utc:
            raise ValueError("local and UTC timestamps must describe the same instant")
        zone_offset = self.timestamp_local.astimezone(
            ZoneInfo(self.observer.timezone)
        ).utcoffset()
        if self.timestamp_local.utcoffset() != zone_offset:
            raise ValueError("timestamp_local offset must match observer timezone")
        return self


class SkyChartRenderMetadata(InputModel):
    projection: Literal["azimuthal_equidistant_zenith"]
    width_px: Literal[2400]
    height_px: Literal[2400]
    layer_order: list[
        Literal[
            "background",
            "horizon_grid",
            "constellations",
            "stars",
            "moon",
            "planets",
            "target",
            "footer",
        ]
    ]
    png_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def require_canonical_layer_order(self) -> "SkyChartRenderMetadata":
        if self.layer_order != _SKY_CHART_LAYER_ORDER:
            raise ValueError("layer_order must use the canonical sky-chart order")
        return self


class SkyChartObjectsMetadata(InputModel):
    moon: SkyChartObject
    planets: list[SkyChartObject]
    target: SkyChartObject | None
    stars_drawn: int = Field(ge=0)
    constellation_segments_drawn: int = Field(ge=0)


class SkyChartCatalogSourceMetadata(InputModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class SkyChartCatalogMetadata(SkyChartCatalogSourceMetadata):
    constellation_segments: SkyChartCatalogSourceMetadata
    status: Literal["available", "degraded"]


class SkyChartCalculationMetadata(InputModel):
    time_scale: Literal["UTC"] = "UTC"
    horizontal_frame: Literal["AltAz"] = "AltAz"
    atmospheric_refraction: Literal[False] = False
    solar_system_ephemeris: Literal["builtin"] = "builtin"
    iers_auto_download: Literal[False] = False


class SkyChartDependenciesMetadata(InputModel):
    python: str = Field(min_length=1)
    astropy: str = Field(min_length=1)
    matplotlib: str = Field(min_length=1)
    tzdata: str = Field(min_length=1)


class SkyChartExportMetadata(InputModel):
    """The complete, non-sensitive JSON export for one rendered PNG."""

    schema_version: Literal["1.0"] = "1.0"
    render_id: str = Field(pattern=_SKY_CHART_RENDER_ID_PATTERN)
    created_at_utc: datetime
    request: SkyChartExportRequest
    render: SkyChartRenderMetadata
    objects: SkyChartObjectsMetadata
    catalog: SkyChartCatalogMetadata
    calculation: SkyChartCalculationMetadata
    dependencies: SkyChartDependenciesMetadata
    warnings: list[str] = Field(default_factory=list)

    @field_validator("created_at_utc")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("created_at_utc must use UTC offset")
        return value

    @field_serializer("created_at_utc", when_used="json")
    def serialize_created_at_utc(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def model_dump_json(self, **kwargs: object) -> str:
        """Serialize export coordinates as six-place JSON numbers, never strings."""
        serialized = super().model_dump_json(**kwargs)
        return _SKY_CHART_COORDINATE_JSON_PATTERN.sub(
            lambda match: f"{match.group(1)}{float(match.group(2)):.6f}",
            serialized,
        )


class TonightRecommendationRequest(InputModel):
    task: ObservationTask
    min_target_altitude_deg: float = Field(default=30.0, ge=-90, le=90)
    max_sun_altitude_deg: float = Field(default=-12.0, ge=-90, le=90)


class RecommendationWindow(InputModel):
    start_local: datetime
    end_local: datetime
    grade: Literal["recommended", "caution", "not_recommended"]
    reasons: list[str] = Field(min_length=1)


class TonightRecommendationResult(InputModel):
    geometry: ObservationPlanResult
    weather_forecast: WeatherForecast
    light_pollution: LightPollutionResult
    recommendations: list[RecommendationWindow]
    human_review: list[str] = Field(min_length=1)
    provenance: list[ExternalSource]


class StellariumSyncRequest(InputModel):
    observer: Observer
    timestamp: datetime
    target: str
