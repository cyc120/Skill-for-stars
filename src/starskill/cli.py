"""Command-line entry point for StarSkill."""

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from starskill.ephemeris_calculator import (
    calculate_ephemeris,
    write_ephemeris_csv,
    write_ephemeris_json,
)
from starskill.observation_planner import (
    plan_observation,
    write_observation_plan_json,
    write_visibility_csv,
)
from starskill.pipeline import run_pipeline
from starskill.public_data_fetcher import (
    PublicDataError,
    PublicDataNotFoundError,
    PublicDataServiceError,
    PublicDataSizeError,
    PublicDataValidationError,
    UrlImageBackend,
    fetch_sdss_image,
    write_public_image_metadata,
)
from starskill.schemas import (
    AstronomicalRelationshipTask,
    EphemerisResult,
    ObservationTask,
    ResolvedTarget,
    SDSSImageRequest,
    SolarSystemRelationshipTask,
    TargetRef,
    VisibilityCriteria,
)
from starskill.solar_system_relationship import (
    calculate_astronomical_relationship,
    calculate_solar_system_relationship,
    write_astronomical_relationship_csv,
    write_relationship_csv,
    write_relationship_json,
)
from starskill.sky_chart_catalog import (
    CatalogDownloadError,
    FullCatalogCache,
    load_hyg_source,
)
from starskill.target_resolver import (
    InvalidTargetNameError,
    SimbadBackend,
    TargetNotFoundError,
    TargetResolutionError,
    TargetServiceError,
    resolve_target,
)
from starskill.target_references import resolve_target_ref
from starskill.visualizer import plot_visibility
from starskill.web_api import HttpCatalogFetcher, run_web_server


class InputValidationError(ValueError):
    """A user-provided input file cannot be parsed as a JSON object."""


TARGET_REF_ADAPTER = TypeAdapter(TargetRef)
RELATIONSHIP_TASK_ADAPTER = TypeAdapter(
    AstronomicalRelationshipTask | SolarSystemRelationshipTask
)
TARGET_BEARING_TASK_ADAPTER = TypeAdapter(
    ObservationTask | AstronomicalRelationshipTask | SolarSystemRelationshipTask
)
TARGET_BEARING_TASK_MODELS = {
    "observation_plan": ObservationTask,
    "astronomical_relationship": AstronomicalRelationshipTask,
    "solar_system_relationship": SolarSystemRelationshipTask,
}


def load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"invalid JSON input: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputValidationError("JSON input must be an object")
    return payload


def require_output_directory(path: Path, flag: str) -> None:
    """Reject an output directory flag that an existing non-directory occupies.

    Without this the first ``mkdir`` raises FileExistsError/NotADirectoryError and
    escapes as a bare traceback, which is neither a documented exit code nor JSON.
    """
    if path.exists() and not path.is_dir():
        raise InputValidationError(f"{flag} must name a directory, but {path} is a file")


def require_output_file(path: Path, flag: str) -> None:
    """Reject an output file flag that an existing directory occupies."""
    if path.is_dir():
        raise InputValidationError(f"{flag} must name a file, but {path} is a directory")


def print_resolution_error(
    exc: TargetResolutionError,
) -> None:
    print(
        json.dumps(
            {"resolved": False, "error": exc.code, "message": str(exc)},
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


def resolution_error_exit_code(exc: TargetResolutionError) -> int:
    if isinstance(exc, InvalidTargetNameError):
        return 2
    if isinstance(exc, TargetNotFoundError):
        return 3
    if isinstance(exc, TargetServiceError):
        return 4
    return 2


def print_validation_error(exc: ValidationError) -> None:
    details = [
        {
            "location": list(error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors(include_url=False, include_context=False)
    ]
    print(
        json.dumps(
            {
                "valid": False,
                "error": "validation_error",
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


def print_input_validation_error(
    exc: InputValidationError, detail_type: str = "json_invalid"
) -> None:
    print(
        json.dumps(
            {
                "valid": False,
                "error": "validation_error",
                "details": [
                    {
                        "location": [],
                        "message": str(exc),
                        "type": detail_type,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


def print_public_data_error(exc: PublicDataError) -> None:
    print(
        json.dumps(
            {"downloaded": False, "error": exc.code, "message": str(exc)},
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )


def download_full_catalog(cache_dir: Path) -> dict[str, object]:
    """Download and publish the one fixed HYG source without exposing its URL."""
    try:
        summary = FullCatalogCache(cache_dir, load_hyg_source()).download_and_publish(
            HttpCatalogFetcher()
        )
    except CatalogDownloadError:
        raise
    except (OSError, ValueError) as error:
        raise CatalogDownloadError("catalog cache setup failed") from error
    return {
        "downloaded": True,
        "version": summary.version,
        "row_count": summary.row_count,
        "compressed_sha256": summary.compressed_sha256,
        "csv_sha256": summary.csv_sha256,
        "cache_status": summary.status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="starskill")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate", help="validate a task JSON file")
    validate_parser.add_argument("input_path", type=Path)

    resolve_parser = commands.add_parser("resolve", help="resolve an astronomy target")
    resolve_parser.add_argument("target")
    resolve_parser.add_argument("--cache-dir", type=Path, default=Path("cache/targets"))
    resolve_parser.add_argument("--output", type=Path)

    resolve_target_parser = commands.add_parser(
        "resolve-target", help="resolve a typed astronomy target reference"
    )
    resolve_target_parser.add_argument("input_path", type=Path)
    resolve_target_parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/targets")
    )
    resolve_target_parser.add_argument("--output", type=Path)

    ephemeris_parser = commands.add_parser(
        "ephemeris", help="calculate target, Sun, and Moon ephemerides"
    )
    ephemeris_parser.add_argument("input_path", type=Path)
    ephemeris_parser.add_argument("--target-file", type=Path)
    ephemeris_parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/targets")
    )
    ephemeris_parser.add_argument("--output", type=Path, required=True)
    ephemeris_parser.add_argument("--metadata", type=Path, required=True)

    plan_parser = commands.add_parser(
        "plan", help="create candidate observation windows and a visibility chart"
    )
    plan_parser.add_argument("ephemeris_path", type=Path)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--metadata", type=Path, required=True)
    plan_parser.add_argument("--figure", type=Path, required=True)
    plan_parser.add_argument("--min-target-altitude-deg", type=float, default=30.0)
    plan_parser.add_argument("--max-sun-altitude-deg", type=float, default=-12.0)

    run_parser = commands.add_parser(
        "run", help="run the complete observation workflow and write an audit bundle"
    )
    run_parser.add_argument("input_path", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--cache-dir", type=Path, default=Path("cache/targets"))
    run_parser.add_argument("--min-target-altitude-deg", type=float, default=30.0)
    run_parser.add_argument("--max-sun-altitude-deg", type=float, default=-12.0)

    relationship_parser = commands.add_parser(
        "relationship",
        help="calculate an apparent astronomical target relationship",
        description="Calculate an apparent astronomical target relationship.",
    )
    relationship_parser.add_argument("input_path", type=Path)
    relationship_parser.add_argument("--output", type=Path, required=True)
    relationship_parser.add_argument("--metadata", type=Path, required=True)
    relationship_parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/targets")
    )

    image_parser = commands.add_parser(
        "fetch-image", help="fetch and process the bounded SDSS DR18 M51 cutout"
    )
    image_parser.add_argument("input_path", type=Path)
    image_parser.add_argument("--output-dir", type=Path, required=True)
    image_parser.add_argument("--cache-dir", type=Path, default=Path("cache/sdss"))

    sky_chart_parser = commands.add_parser(
        "sky-chart", help="start the local Python sky chart"
    )
    sky_chart_parser.add_argument("--port", type=int, default=8000)
    sky_chart_parser.add_argument("--open", action="store_true")
    sky_chart_parser.add_argument("--download-catalog", action="store_true")
    sky_chart_parser.add_argument(
        "--catalog-cache-dir", type=Path, default=Path("cache/sky-chart")
    )

    args = parser.parse_args(argv)

    if args.command == "sky-chart":
        if not 1024 <= args.port <= 65535:
            parser.error("--port must be between 1024 and 65535")
        if args.download_catalog:
            try:
                summary = download_full_catalog(args.catalog_cache_dir)
            except CatalogDownloadError:
                print(
                    json.dumps(
                        {"downloaded": False, "error": "catalog_download_failed"},
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
                return 1
            print(json.dumps(summary, ensure_ascii=False))
            return 0
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                run_web_server(
                    port=args.port,
                    open_browser=args.open,
                    catalog_cache_dir=args.catalog_cache_dir,
                )
        except SystemExit as error:
            if error.code != 1:
                raise
        except (OSError, RuntimeError):
            pass
        else:
            return 0
        print(
            json.dumps(
                {"started": False, "error": "web_server_start_failed"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    if args.command == "resolve":
        try:
            if args.output:
                require_output_file(args.output, "--output")
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        try:
            target = resolve_target(
                args.target,
                backend=SimbadBackend(),
                cache_dir=args.cache_dir,
            )
        except InvalidTargetNameError as exc:
            print_resolution_error(exc)
            return 2
        except TargetNotFoundError as exc:
            print_resolution_error(exc)
            return 3
        except TargetServiceError as exc:
            print_resolution_error(exc)
            return 4
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(target.model_dump_json(indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"resolved": True, "target": target.model_dump(mode="json")},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "resolve-target":
        try:
            reference = TARGET_REF_ADAPTER.validate_python(
                load_json_object(args.input_path)
            )
        except InputValidationError as exc:
            print_input_validation_error(exc)
            return 2
        except ValidationError as exc:
            print_validation_error(exc)
            return 2
        try:
            if args.output:
                require_output_file(args.output, "--output")
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        try:
            target = resolve_target_ref(
                reference,
                backend=SimbadBackend() if reference.kind == "simbad" else None,
                cache_dir=args.cache_dir,
            )
        except (InvalidTargetNameError, TargetResolutionError) as exc:
            print_resolution_error(exc)
            return resolution_error_exit_code(exc)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(target.model_dump_json(indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"resolved": True, "target": target.model_dump(mode="json")},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "plan":
        try:
            ephemeris = EphemerisResult.model_validate(
                load_json_object(args.ephemeris_path)
            )
            criteria = VisibilityCriteria(
                min_target_altitude_deg=args.min_target_altitude_deg,
                max_sun_altitude_deg=args.max_sun_altitude_deg,
            )
        except InputValidationError as exc:
            # Only a real JSON parse failure reaches here, so json_invalid is right.
            print_input_validation_error(exc)
            return 2
        except ValidationError as exc:
            print_validation_error(exc)
            return 2
        try:
            require_output_file(args.output, "--output")
            require_output_file(args.metadata, "--metadata")
            require_output_file(args.figure, "--figure")
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        plan = plan_observation(ephemeris, criteria)
        write_visibility_csv(plan, args.output)
        write_observation_plan_json(plan, args.metadata)
        plot_visibility(plan, args.figure)
        print(
            json.dumps(
                {
                    "planned": True,
                    "sample_count": len(plan.samples),
                    "window_count": len(plan.windows),
                    "csv": str(args.output),
                    "metadata": str(args.metadata),
                    "figure": str(args.figure),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "relationship":
        try:
            payload = load_json_object(args.input_path)
        except InputValidationError as exc:
            print_input_validation_error(exc)
            return 2
        try:
            relationship_task = RELATIONSHIP_TASK_ADAPTER.validate_python(payload)
            require_output_file(args.output, "--output")
            require_output_file(args.metadata, "--metadata")
        except ValidationError as exc:
            print_validation_error(exc)
            return 2
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        if isinstance(relationship_task, SolarSystemRelationshipTask):
            result = calculate_solar_system_relationship(relationship_task)
            write_relationship_csv(result, args.output)
        else:
            try:
                result = calculate_astronomical_relationship(
                    relationship_task,
                    target_backend=(
                        SimbadBackend()
                        if "simbad"
                        in {relationship_task.primary.kind, relationship_task.secondary.kind}
                        else None
                    ),
                    cache_dir=args.cache_dir,
                )
            except (InvalidTargetNameError, TargetResolutionError) as exc:
                print_resolution_error(exc)
                return resolution_error_exit_code(exc)
            write_astronomical_relationship_csv(result, args.output)
        write_relationship_json(result, args.metadata)
        separations = [sample.angular_separation_deg for sample in result.samples]
        print(
            json.dumps(
                {
                    "calculated": True,
                    "sample_count": len(result.samples),
                    "minimum_separation_deg": min(separations),
                    "maximum_separation_deg": max(separations),
                    "csv": str(args.output),
                    "metadata": str(args.metadata),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "fetch-image":
        try:
            payload = load_json_object(args.input_path)
        except InputValidationError as exc:
            print_input_validation_error(exc)
            return 2
        try:
            image_request = SDSSImageRequest.model_validate(payload)
            require_output_directory(args.output_dir, "--output-dir")
        except ValidationError as exc:
            print_validation_error(exc)
            return 2
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        try:
            result = fetch_sdss_image(
                image_request,
                cache_dir=args.cache_dir,
                source_path=args.output_dir / "data/m51_sdss.jpg",
                display_path=args.output_dir / "figures/m51_display.png",
                backend=UrlImageBackend(),
            )
        except PublicDataNotFoundError as exc:
            print_public_data_error(exc)
            return 6
        except PublicDataServiceError as exc:
            print_public_data_error(exc)
            return 7
        except PublicDataSizeError as exc:
            print_public_data_error(exc)
            return 8
        except PublicDataValidationError as exc:
            print_public_data_error(exc)
            return 9
        metadata_path = args.output_dir / "image_metadata.json"
        write_public_image_metadata(result, metadata_path)
        print(
            json.dumps(
                {
                    "downloaded": True,
                    "from_cache": result.source.from_cache,
                    "source_image": result.source_path,
                    "display_image": result.display_path,
                    "metadata": str(metadata_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        payload = load_json_object(args.input_path)
    except InputValidationError as exc:
        print_input_validation_error(exc)
        return 2
    if args.command == "validate":
        discriminator = payload.get("task_type", "observation_plan")
        task_model = (
            TARGET_BEARING_TASK_MODELS.get(discriminator)
            if isinstance(discriminator, str)
            else None
        )
        if task_model is None:
            try:
                TARGET_BEARING_TASK_ADAPTER.validate_python(payload)
            except ValidationError as exc:
                print_validation_error(exc)
                return 2
            raise AssertionError("unreachable task discriminator")
        try:
            task = task_model.model_validate(payload)
        except ValidationError as exc:
            print_validation_error(exc)
            return 2
        print(
            json.dumps(
                {"valid": True, "task": task.model_dump(mode="json")},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        task = ObservationTask.model_validate(payload)
    except ValidationError as exc:
        print_validation_error(exc)
        return 2

    if args.command == "ephemeris":
        try:
            require_output_file(args.output, "--output")
            require_output_file(args.metadata, "--metadata")
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        if args.target_file is not None:
            try:
                target = ResolvedTarget.model_validate(
                    load_json_object(args.target_file)
                )
            except InputValidationError as exc:
                print_input_validation_error(exc)
                return 2
            except ValidationError as exc:
                print_validation_error(exc)
                return 2
        else:
            try:
                target = resolve_target_ref(
                    task.target,
                    backend=(
                        SimbadBackend() if task.target.kind == "simbad" else None
                    ),
                    cache_dir=args.cache_dir,
                )
            except (InvalidTargetNameError, TargetResolutionError) as exc:
                print_resolution_error(exc)
                return resolution_error_exit_code(exc)
        result = calculate_ephemeris(task, target)
        write_ephemeris_csv(result, args.output)
        write_ephemeris_json(result, args.metadata)
        print(
            json.dumps(
                {
                    "calculated": True,
                    "sample_count": len(result.samples),
                    "csv": str(args.output),
                    "metadata": str(args.metadata),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run":
        try:
            criteria = VisibilityCriteria(
                min_target_altitude_deg=args.min_target_altitude_deg,
                max_sun_altitude_deg=args.max_sun_altitude_deg,
            )
            require_output_directory(args.output_dir, "--output-dir")
        except ValidationError as exc:
            print_validation_error(exc)
            return 2
        except InputValidationError as exc:
            print_input_validation_error(exc, "value_error")
            return 2
        try:
            outcome = run_pipeline(
                task,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                backend=SimbadBackend(),
                criteria=criteria,
            )
        except TargetResolutionError as exc:
            # Catch the base class: an unsupported solar-system body is neither a
            # not-found nor a service failure, and catching only the three named
            # subclasses let it escape as a bare traceback with exit 1.
            print_resolution_error(exc)
            return resolution_error_exit_code(exc)
        print(
            json.dumps(
                {
                    "status": outcome.status,
                    "run_id": outcome.manifest.run_id,
                    "cache_hit": outcome.manifest.cache_hit,
                    "output_dir": outcome.output_dir,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if outcome.status == "success" else 5

    raise AssertionError(f"unhandled command: {args.command}")
