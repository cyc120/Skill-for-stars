import csv
import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

import starskill.cli as cli
import starskill.target_resolver as target_resolver
from starskill.cli import main
from starskill.sky_chart_catalog import CatalogDownloadError
from tests.fixtures.evaluation.replay_fixtures import write_fixed_m31_cache
from tests.fixtures.m42 import write_m42_ephemeris, write_m42_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_python_only_sky_chart() -> None:
    runtime_documents = {
        "README": PROJECT_ROOT / "README.md",
        "Skill": PROJECT_ROOT / "skills" / "run-starskill" / "SKILL.md",
        "CLI reference": PROJECT_ROOT
        / "skills"
        / "run-starskill"
        / "references"
        / "cli-contract.md",
    }
    document_text = {
        name: path.read_text(encoding="utf-8").lower()
        for name, path in runtime_documents.items()
    }
    text = document_text["README"]
    hyg_source = (PROJECT_ROOT / "docs" / "sources" / "hyg-v4.1.md").read_text(
        encoding="utf-8"
    )

    assert 'pip install ".[dev]"' in text
    assert 'pip install -e ".[dev]"' not in text
    assert "starskill sky-chart --open" in text
    assert ".venv/bin/starskill sky-chart --open" in document_text["Skill"]
    assert text.count("```") % 2 == 0
    assert "```text\nverified 100 packaged HYG v4.1 records\n```" in hyg_source
    assert "packaged HYG v4.1 records against /tmp" not in hyg_source
    for name, value in document_text.items():
        for legacy_dependency in (
            r"\bnode(?:\.js)?\b",
            r"\bnpm\b",
            r"\bdocker\b",
            r"\bmake\b",
            r"web/dist",
            r"stellarium web engine",
        ):
            assert re.search(legacy_dependency, value) is None, (
                f"{name} contains legacy browser dependency {legacy_dependency!r}"
            )


def test_run_starskill_skill_requires_direct_visual_delivery() -> None:
    skill = (
        PROJECT_ROOT / "skills" / "run-starskill" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "## Deliver Generated Results" in skill
    assert "Embed each verified user-relevant raster artifact with Markdown" in skill
    assert "require the JSON's `render.png_sha256` to match the saved PNG" in skill
    assert "center is the zenith, the outer circle is the horizon" in skill
    assert "do not invent one" in skill


def test_public_docs_distinguish_script_owned_acceptance_from_agent_evaluation() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    evaluation = (PROJECT_ROOT / "evaluation" / "README.md").read_text(encoding="utf-8")
    contract = (
        PROJECT_ROOT / "skills" / "run-starskill" / "references" / "cli-contract.md"
    ).read_text(encoding="utf-8")

    assert "scripts/evaluate_starskill.py acceptance" in readme
    assert "--run-root" in readme
    assert "script_owned_engineering_acceptance" in evaluation
    assert "does not replace external Worker or Reviewer evidence" in evaluation
    assert "evidence_mode: script_owned_engineering" in evaluation
    assert "execution.json" in contract


def test_changed_public_documents_have_no_trailing_whitespace() -> None:
    public_documents = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "mcp-server.md",
        PROJECT_ROOT / "docs" / "sources" / "hyg-v4.1.md",
        PROJECT_ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-07-23-live-outreach-design.md",
        PROJECT_ROOT / "skills" / "run-starskill" / "SKILL.md",
        PROJECT_ROOT / "skills" / "run-starskill" / "references" / "cli-contract.md",
    )

    for path in public_documents:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert line == line.rstrip(" \t"), f"{path}:{line_number} has trailing whitespace"


def source_checkout_environment() -> dict[str, str]:
    """Make the source package importable to the real child CLI process."""
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, existing_pythonpath) if path
    )
    return environment


class CliSimbadBackend:
    service_url = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

    def query_object(self, query_name: str) -> dict:
        return {
            "canonical_name": "M 42",
            "ra_deg": 83.822083,
            "dec_deg": -5.391111,
            "object_type": "HII",
            "aliases": ["M 42", "NGC 1976", "Orion Nebula"],
        }


class CliEmptySimbadBackend:
    service_url = CliSimbadBackend.service_url

    def query_object(self, query_name: str) -> None:
        return None


class CliFailingSimbadBackend:
    service_url = CliSimbadBackend.service_url

    def query_object(self, query_name: str) -> dict:
        raise TimeoutError("SIMBAD request timed out")


class CliImageBackend:
    def fetch(self, url: str, *, timeout_seconds: int, max_bytes: int) -> tuple[bytes, str]:
        image = Image.new("RGB", (512, 512), "#101820")
        draw = ImageDraw.Draw(image)
        draw.ellipse((100, 80, 410, 430), fill="#d9e5f2")
        output = BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue(), "image/jpeg"


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def coordinate_relationship_task() -> dict[str, object]:
    return {
        "task_type": "astronomical_relationship",
        "primary": {"kind": "coordinates", "label": "A", "ra_deg": 10, "dec_deg": 20},
        "secondary": {"kind": "coordinates", "label": "B", "ra_deg": 11, "dec_deg": 21},
        "observer": {
            "location_name": "Shanghai",
            "longitude": 121.4737,
            "latitude": 31.2304,
            "timezone": "Asia/Shanghai",
        },
        "time_range": {
            "start": "2026-01-10T18:00:00+08:00",
            "end": "2026-01-10T18:20:00+08:00",
        },
        "interval_minutes": 20,
    }


def coordinate_observation_task() -> dict[str, object]:
    return {
        "task_type": "observation_plan",
        "target": {"kind": "coordinates", "label": "A", "ra_deg": 10, "dec_deg": 20},
        "observer": {
            "location_name": "Shanghai",
            "longitude": 121.4737,
            "latitude": 31.2304,
            "timezone": "Asia/Shanghai",
        },
        "time_range": {
            "start": "2026-01-10T18:00:00+08:00",
            "end": "2026-01-10T18:20:00+08:00",
        },
        "interval_minutes": 20,
    }


def test_validate_command_prints_canonical_task(tmp_path, capsys) -> None:
    input_path = tmp_path / "task.json"
    input_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", str(input_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["valid"] is True
    assert output["task"]["target"] == {"kind": "simbad", "name": "M42"}
    assert output["task"]["interval_minutes"] == 10


def test_validate_command_returns_structured_validation_errors(
    tmp_path, capsys
) -> None:
    input_path = tmp_path / "invalid-task.json"
    input_path.write_text(
        json.dumps(
            {
                "task_type": "observation_plan",
                "target": "M42",
                "observer": {
                    "location_name": "Beijing",
                    "longitude": 116.4074,
                    "latitude": 39.9042,
                    "timezone": "Mars/Olympus_Mons",
                },
                "time_range": {
                    "start": "2026-01-10 18:00:00",
                    "end": "2026-01-11 02:00:00",
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", str(input_path)])
    captured = capsys.readouterr()
    output = json.loads(captured.err)

    assert exit_code == 2
    assert output["valid"] is False
    assert output["error"] == "validation_error"
    assert output["details"][0]["location"] == ["observer", "timezone"]


def test_validate_command_returns_structured_error_for_unhashable_task_type(
    tmp_path: Path, capsys
) -> None:
    input_path = write_json(tmp_path / "invalid-task-type.json", {"task_type": []})

    exit_code = main(["validate", str(input_path)])
    output = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert output["valid"] is False
    assert output["error"] == "validation_error"


@pytest.mark.parametrize(
    "payload",
    [
        coordinate_relationship_task(),
        json.loads((PROJECT_ROOT / "examples/moon_jupiter_shanghai.json").read_text()),
    ],
)
def test_validate_command_accepts_relationship_task_versions(
    tmp_path: Path, capsys, payload: dict[str, object]
) -> None:
    input_path = write_json(tmp_path / "task.json", payload)

    exit_code = main(["validate", str(input_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["valid"] is True
    assert output["task"]["task_type"] == payload["task_type"]


@pytest.mark.parametrize("command", ["validate", "relationship", "fetch-image", "run"])
def test_cli_rejects_malformed_json_with_structured_error(
    tmp_path, capsys, command
) -> None:
    input_path = tmp_path / "malformed.json"
    input_path.write_text("{not-json", encoding="utf-8")
    argv = {
        "validate": ["validate", str(input_path)],
        "relationship": [
            "relationship",
            str(input_path),
            "--output",
            str(tmp_path / "relationship.csv"),
            "--metadata",
            str(tmp_path / "relationship.json"),
        ],
        "fetch-image": [
            "fetch-image",
            str(input_path),
            "--output-dir",
            str(tmp_path / "image"),
        ],
        "run": [
            "run",
            str(input_path),
            "--output-dir",
            str(tmp_path / "run"),
        ],
    }[command]

    exit_code = main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 2
    assert payload["valid"] is False
    assert payload["error"] == "validation_error"
    assert payload["details"][0]["type"] == "json_invalid"
    assert "Traceback" not in captured.err


def test_ephemeris_rejects_malformed_target_file_with_structured_error(
    tmp_path, capsys
) -> None:
    target_path = tmp_path / "malformed-target.json"
    target_path.write_text("{not-json", encoding="utf-8")
    output_path = tmp_path / "ephemeris.csv"
    metadata_path = tmp_path / "ephemeris.json"

    exit_code = main(
        [
            "ephemeris",
            str(PROJECT_ROOT / "examples/observation_m42_beijing.json"),
            "--target-file",
            str(target_path),
            "--output",
            str(output_path),
            "--metadata",
            str(metadata_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 2
    assert payload["valid"] is False
    assert payload["error"] == "validation_error"
    assert payload["details"][0]["type"] == "json_invalid"
    assert not output_path.exists()
    assert not metadata_path.exists()


def test_python_module_entrypoint_validates_documented_example() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "starskill",
            "validate",
            "examples/observation_m42_beijing.json",
        ],
        cwd=PROJECT_ROOT,
        env=source_checkout_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"] is True


def test_module_help_lists_resolve_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "starskill", "--help"],
        cwd=PROJECT_ROOT,
        env=source_checkout_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "resolve" in result.stdout


def test_module_help_lists_ephemeris_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "starskill", "--help"],
        cwd=PROJECT_ROOT,
        env=source_checkout_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ephemeris" in result.stdout


def test_ephemeris_command_writes_csv_and_json(tmp_path, capsys) -> None:
    csv_path = tmp_path / "day3" / "intermediate" / "ephemeris.csv"
    json_path = tmp_path / "day3" / "intermediate" / "ephemeris.json"
    target_path = tmp_path / "day2" / "intermediate" / "target_resolved.json"
    write_m42_target(target_path)

    exit_code = main(
        [
            "ephemeris",
            str(PROJECT_ROOT / "examples/observation_m42_beijing.json"),
            "--target-file",
            str(target_path),
            "--output",
            str(csv_path),
            "--metadata",
            str(json_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["calculated"] is True
    assert summary["sample_count"] == 49
    assert len(rows) == 49
    assert len(metadata["samples"]) == 49


def test_run_reports_an_unsupported_solar_system_body_as_validation_error(
    tmp_path: Path, capsys
) -> None:
    task_path = write_json(
        tmp_path / "task.json",
        {
            "task_type": "observation_plan",
            "target": {"kind": "solar_system", "body": "pluto"},
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
        },
    )
    output_dir = tmp_path / "out"

    exit_code = main(["run", str(task_path), "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "unsupported_solar_system_body"
    # The failed run still records its own structured evidence.
    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["issues"][0]["code"] == "unsupported_solar_system_body"


def test_output_flags_reject_a_path_an_existing_entry_occupies(
    tmp_path: Path, capsys
) -> None:
    occupied_file = tmp_path / "occupied"
    occupied_file.write_text("not a directory", encoding="utf-8")
    occupied_dir = tmp_path / "occupied-dir"
    occupied_dir.mkdir()
    free = tmp_path / "free"
    task = str(PROJECT_ROOT / "examples/observation_m42_beijing.json")
    ephemeris_path = tmp_path / "ephemeris.json"
    write_m42_ephemeris(ephemeris_path)

    cases = [
        (["run", task, "--output-dir", str(occupied_file)], "--output-dir"),
        (
            [
                "fetch-image",
                str(PROJECT_ROOT / "examples/m51_sdss_image.json"),
                "--output-dir",
                str(occupied_file),
            ],
            "--output-dir",
        ),
        (
            [
                "relationship",
                str(PROJECT_ROOT / "examples/relationships/mars_m31.json"),
                "--output",
                str(occupied_dir),
                "--metadata",
                str(free / "relationship.json"),
            ],
            "--output",
        ),
        (
            ["ephemeris", task, "--output", str(occupied_dir), "--metadata", str(free / "e.json")],
            "--output",
        ),
        (
            [
                "plan",
                str(ephemeris_path),
                "--output",
                str(free / "visibility.csv"),
                "--metadata",
                str(free / "result.json"),
                "--figure",
                str(occupied_dir),
            ],
            "--figure",
        ),
        (["resolve", "M42", "--output", str(occupied_dir)], "--output"),
    ]

    for argv, flag in cases:
        assert main(argv) == 2, argv
        captured = capsys.readouterr()
        payload = json.loads(captured.err)
        assert payload["error"] == "validation_error"
        assert flag in payload["details"][0]["message"]
        assert captured.out == ""
    assert not free.exists()


def test_resolve_target_and_ephemeris_accept_typed_references(
    tmp_path: Path, capsys
) -> None:
    reference_path = write_json(
        tmp_path / "mars.json", {"kind": "solar_system", "body": "mars"}
    )
    task_path = write_json(
        tmp_path / "coordinate-observation.json", coordinate_observation_task()
    )
    resolved_path = tmp_path / "mars-resolved.json"
    ephemeris_path = tmp_path / "ephemeris.csv"
    metadata_path = tmp_path / "ephemeris.json"

    assert main(
        ["resolve-target", str(reference_path), "--output", str(resolved_path)]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "ephemeris",
            str(task_path),
            "--output",
            str(ephemeris_path),
            "--metadata",
            str(metadata_path),
        ]
    ) == 0

    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    ephemeris = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert resolved["motion"] == "dynamic"
    assert resolved["source"]["provider"] == "astropy_builtin_ephemeris"
    assert ephemeris["target"]["kind"] == "coordinates"
    assert ephemeris["target"]["source"]["provider"] == "user_coordinates"


@pytest.mark.parametrize(
    ("command", "payload", "argv_suffix"),
    [
        (
            "resolve-target",
            {"kind": "simbad", "name": "M42; SELECT *"},
            [],
        ),
        (
            "ephemeris",
            {
                **coordinate_observation_task(),
                "target": {"kind": "simbad", "name": "M42; SELECT *"},
            },
            ["--output", "ephemeris.csv", "--metadata", "ephemeris.json"],
        ),
        (
            "relationship",
            {
                **coordinate_relationship_task(),
                "primary": {"kind": "simbad", "name": "M42; SELECT *"},
            },
            ["--output", "relationship.csv", "--metadata", "relationship.json"],
        ),
    ],
)
def test_typed_simbad_invalid_name_returns_structured_error(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    payload: dict[str, object],
    argv_suffix: list[str],
) -> None:
    class CapabilityTrackingSimbadClient:
        def __init__(self) -> None:
            self.capability_requests: list[tuple[str, ...]] = []

        def add_votable_fields(self, *fields: str) -> None:
            self.capability_requests.append(fields)

    client = CapabilityTrackingSimbadClient()
    monkeypatch.setattr(target_resolver, "Simbad", lambda: client)
    input_path = write_json(tmp_path / "invalid-simbad.json", payload)
    argv = [command, str(input_path)]
    for argument in argv_suffix:
        argv.append(
            str(tmp_path / argument)
            if argument.endswith(".json") or argument.endswith(".csv")
            else argument
        )

    exit_code = main(argv)
    output = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert output == {
        "resolved": False,
        "error": "invalid_target_name",
        "message": "target name contains unsafe characters",
    }
    assert client.capability_requests == []


@pytest.mark.parametrize(
    ("command", "payload", "output_arguments"),
    [
        ("resolve-target", {"kind": "simbad", "name": "M31"}, []),
        (
            "ephemeris",
            {
                **coordinate_observation_task(),
                "target": {"kind": "simbad", "name": "M31"},
            },
            ["--output", "ephemeris.csv", "--metadata", "ephemeris.json"],
        ),
        (
            "relationship",
            {
                **coordinate_relationship_task(),
                "primary": {"kind": "simbad", "name": "M31"},
            },
            ["--output", "relationship.csv", "--metadata", "relationship.json"],
        ),
    ],
)
def test_cached_typed_simbad_commands_never_touch_client_capabilities_or_query(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    payload: dict[str, object],
    output_arguments: list[str],
) -> None:
    class OfflineTrackingSimbadClient:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        @property
        def timeout(self) -> int | None:
            return None

        @timeout.setter
        def timeout(self, value: int) -> None:
            self.events.append(("timeout", value))

        @property
        def ROW_LIMIT(self) -> int | None:
            return None

        @ROW_LIMIT.setter
        def ROW_LIMIT(self, value: int) -> None:
            self.events.append(("row_limit", value))

        def add_votable_fields(self, *fields: str) -> None:
            self.events.append(("fields", fields))

        def query_object(self, query_name: str) -> None:
            self.events.append(("query", query_name))
            pytest.fail("a valid deterministic cache hit must remain offline")

    client = OfflineTrackingSimbadClient()
    monkeypatch.setattr(target_resolver, "Simbad", lambda: client)
    cache_dir = tmp_path / "target-cache"
    write_fixed_m31_cache(cache_dir)
    input_path = write_json(tmp_path / f"{command}.json", payload)
    argv = [command, str(input_path)]
    for argument in output_arguments:
        argv.append(
            str(tmp_path / argument)
            if argument.endswith(".json") or argument.endswith(".csv")
            else argument
        )
    argv.extend(["--cache-dir", str(cache_dir)])

    exit_code = main(argv)
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert client.events == []


def test_module_help_lists_plan_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "starskill", "--help"],
        cwd=PROJECT_ROOT,
        env=source_checkout_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "plan" in result.stdout


def test_relationship_help_describes_apparent_astronomical_relationship(capsys) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["relationship", "--help"])

    assert help_exit.value.code == 0
    assert "apparent astronomical target relationship" in capsys.readouterr().out


def test_plan_command_writes_visibility_result_and_figure(tmp_path, capsys) -> None:
    csv_path = tmp_path / "day4" / "intermediate" / "visibility.csv"
    json_path = tmp_path / "day4" / "result.json"
    figure_path = tmp_path / "day4" / "figures" / "visibility_curve.png"
    ephemeris_path = tmp_path / "day3" / "intermediate" / "ephemeris.json"
    write_m42_ephemeris(ephemeris_path)

    exit_code = main(
        [
            "plan",
            str(ephemeris_path),
            "--output",
            str(csv_path),
            "--metadata",
            str(json_path),
            "--figure",
            str(figure_path),
            "--min-target-altitude-deg",
            "30",
            "--max-sun-altitude-deg",
            "-12",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["planned"] is True
    assert summary["sample_count"] == 49
    assert summary["window_count"] == 1
    assert csv_path.is_file()
    assert figure_path.is_file()
    assert len(metadata["windows"]) == 1


def test_plan_command_rejects_invalid_threshold_without_outputs(
    tmp_path, capsys
) -> None:
    csv_path = tmp_path / "visibility.csv"
    json_path = tmp_path / "result.json"
    figure_path = tmp_path / "visibility.png"
    ephemeris_path = tmp_path / "day3" / "intermediate" / "ephemeris.json"
    write_m42_ephemeris(ephemeris_path)

    try:
        exit_code = main(
            [
                "plan",
                str(ephemeris_path),
                "--output",
                str(csv_path),
                "--metadata",
                str(json_path),
                "--figure",
                str(figure_path),
                "--min-target-altitude-deg",
                "100",
            ]
        )
    except ValidationError:
        pytest.fail("plan command leaked a Pydantic ValidationError")

    error = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert error["error"] == "validation_error"
    assert error["details"][0]["location"] == ["min_target_altitude_deg"]
    assert not csv_path.exists()
    assert not json_path.exists()
    assert not figure_path.exists()


def test_run_command_executes_complete_pipeline(tmp_path, capsys, monkeypatch) -> None:
    assert hasattr(cli, "run_pipeline"), "CLI pipeline runner is missing"
    monkeypatch.setattr(cli, "SimbadBackend", CliSimbadBackend)
    output_dir = tmp_path / "complete-run"

    exit_code = main(
        [
            "run",
            str(PROJECT_ROOT / "examples/observation_m42_beijing.json"),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["status"] == "success"
    assert manifest["status"] == "success"
    assert (output_dir / "result.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "review_checklist.md").is_file()


def test_run_command_returns_service_error_and_failed_manifest(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "SimbadBackend", CliFailingSimbadBackend)
    output_dir = tmp_path / "failed-run"

    exit_code = main(
        [
            "run",
            str(PROJECT_ROOT / "examples/observation_m42_beijing.json"),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    error = json.loads(capsys.readouterr().err)
    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert exit_code == 4
    assert error["error"] == "target_service_error"
    assert manifest["status"] == "failed"
    assert not (output_dir / "result.json").exists()


def test_relationship_command_writes_csv_and_json(tmp_path, capsys) -> None:
    input_path = tmp_path / "relationship-task.json"
    input_path.write_text(
        json.dumps(
            {
                "task_type": "solar_system_relationship",
                "targets": ["moon", "jupiter"],
                "observer": {
                    "location_name": "Shanghai",
                    "longitude": 121.4737,
                    "latitude": 31.2304,
                    "timezone": "Asia/Shanghai",
                },
                "time_range": {
                    "start": "2026-03-20 19:00:00",
                    "end": "2026-03-20 23:00:00",
                },
                "interval_minutes": 20,
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "relationship.csv"
    json_path = tmp_path / "relationship.json"

    exit_code = main(
        [
            "relationship",
            str(input_path),
            "--output",
            str(csv_path),
            "--metadata",
            str(json_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["calculated"] is True
    assert summary["sample_count"] == 13
    assert summary["maximum_separation_deg"] > summary["minimum_separation_deg"]
    assert csv_path.is_file()
    assert len(payload["samples"]) == 13
    assert "schema_version" not in payload["settings"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert "moon_altitude_deg" in next(csv.reader(handle))


def test_relationship_cli_accepts_generic_coordinate_pair(
    tmp_path: Path, capsys
) -> None:
    input_path = write_json(tmp_path / "task.json", coordinate_relationship_task())
    csv_path = tmp_path / "relationship.csv"
    metadata_path = tmp_path / "relationship.json"

    exit_code = main(
        [
            "relationship",
            str(input_path),
            "--output",
            str(csv_path),
            "--metadata",
            str(metadata_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        columns = next(csv.reader(handle))
    assert exit_code == 0
    assert summary["sample_count"] == 2
    assert metadata["settings"]["schema_version"] == "2.0"
    assert "primary_altitude_deg" in columns
    assert "moon_altitude_deg" not in columns


@pytest.mark.parametrize(
    ("filename", "primary_kind", "secondary_kind"),
    [
        ("mars_saturn.json", "solar_system", "solar_system"),
        ("mars_m31.json", "solar_system", "simbad"),
        ("m31_coordinates.json", "simbad", "coordinates"),
        ("coordinates_coordinates.json", "coordinates", "coordinates"),
    ],
)
def test_fixed_v2_relationship_examples_validate(
    filename: str, primary_kind: str, secondary_kind: str, capsys
) -> None:
    input_path = PROJECT_ROOT / "examples" / "relationships" / filename

    exit_code = main(["validate", str(input_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["task"]["primary"]["kind"] == primary_kind
    assert output["task"]["secondary"]["kind"] == secondary_kind


def test_fetch_image_command_writes_source_display_and_metadata(
    tmp_path, capsys, monkeypatch
) -> None:
    assert hasattr(cli, "UrlImageBackend"), "CLI image backend is missing"
    monkeypatch.setattr(cli, "UrlImageBackend", CliImageBackend)
    request_path = tmp_path / "m51-request.json"
    request_path.write_text(
        json.dumps(
            {
                "target_name": "M51",
                "data_release": "DR18",
                "ra_deg": 202.4696,
                "dec_deg": 47.1952,
                "scale_arcsec_per_pixel": 0.396,
                "width": 512,
                "height": 512,
                "timeout_seconds": 30,
                "max_bytes": 5000000,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "m51-run"

    exit_code = main(
        [
            "fetch-image",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    metadata = json.loads(
        (output_dir / "image_metadata.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert summary["downloaded"] is True
    assert summary["from_cache"] is False
    assert (output_dir / "data/m51_sdss.jpg").is_file()
    assert (output_dir / "figures/m51_display.png").is_file()
    assert metadata["source"]["database"] == "SDSS SkyServer"


def test_resolve_command_prints_structured_target(
    tmp_path, capsys, monkeypatch
) -> None:
    assert hasattr(cli, "SimbadBackend"), "CLI SIMBAD backend is missing"
    monkeypatch.setattr(cli, "SimbadBackend", CliSimbadBackend)
    output_path = tmp_path / "run" / "intermediate" / "target_resolved.json"

    try:
        exit_code = main(
            [
                "resolve",
                "猎户座大星云",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output",
                str(output_path),
            ]
        )
    except SystemExit:
        pytest.fail("resolve command does not accept --output")
    output = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output["resolved"] is True
    assert output["target"]["query_name"] == "M 42"
    assert output["target"]["canonical_name"] == "M 42"
    assert output["target"]["coordinate_frame"] == "ICRS"
    assert written["canonical_name"] == "M 42"
    assert written["source"]["database"] == "SIMBAD"


def test_resolve_command_returns_structured_not_found_error(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "SimbadBackend", CliEmptySimbadBackend)

    try:
        exit_code = main(
            ["resolve", "Unknown Object", "--cache-dir", str(tmp_path)]
        )
    except Exception as exc:
        pytest.fail(f"CLI leaked {type(exc).__name__} instead of structured JSON")
    output = json.loads(capsys.readouterr().err)

    assert exit_code == 3
    assert output["resolved"] is False
    assert output["error"] == "target_not_found"


def test_resolve_command_returns_structured_invalid_name_error(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "SimbadBackend", CliSimbadBackend)

    try:
        exit_code = main(
            ["resolve", "M42; SELECT *", "--cache-dir", str(tmp_path)]
        )
    except Exception as exc:
        pytest.fail(f"CLI leaked {type(exc).__name__} instead of structured JSON")
    output = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert output["resolved"] is False
    assert output["error"] == "invalid_target_name"


def test_resolve_command_returns_structured_service_error(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "SimbadBackend", CliFailingSimbadBackend)

    try:
        exit_code = main(["resolve", "M42", "--cache-dir", str(tmp_path)])
    except Exception as exc:
        pytest.fail(f"CLI leaked {type(exc).__name__} instead of structured JSON")
    output = json.loads(capsys.readouterr().err)

    assert exit_code == 4
    assert output["resolved"] is False
    assert output["error"] == "target_service_error"


def test_sky_chart_help_and_port_bounds(capsys) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["sky-chart", "--help"])
    assert help_exit.value.code == 0
    help_output = capsys.readouterr().out
    assert set(re.findall(r"--[a-z-]+", help_output)) == {
        "--help",
        "--port",
        "--open",
        "--download-catalog",
        "--catalog-cache-dir",
    }

    for port in ("1023", "65536"):
        with pytest.raises(SystemExit) as port_exit:
            main(["sky-chart", "--port", port])
        assert port_exit.value.code == 2
        capsys.readouterr()

    for forbidden_args in (
        ["--host", "0.0.0.0"],
        ["--source-url", "https://example.test/catalog.csv.gz"],
        ["--network"],
        ["--browser-engine", "firefox"],
        ["https://example.test/catalog.csv.gz"],
    ):
        with pytest.raises(SystemExit) as forbidden_exit:
            main(["sky-chart", *forbidden_args])
        assert forbidden_exit.value.code == 2
        capsys.readouterr()


def test_download_catalog_does_not_start_web_server(
    tmp_path, monkeypatch, capsys
) -> None:
    started = []
    monkeypatch.setattr(cli, "run_web_server", lambda **kwargs: started.append(kwargs))
    monkeypatch.setattr(
        cli,
        "download_full_catalog",
        lambda cache_dir: {
            "downloaded": True,
            "version": "4.1",
            "row_count": 100001,
            "compressed_sha256": "a" * 64,
            "csv_sha256": "b" * 64,
            "cache_status": "available",
        },
    )

    assert (
        main(
            [
                "sky-chart",
                "--download-catalog",
                "--catalog-cache-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["downloaded"] is True
    assert started == []


def test_download_catalog_failure_is_stable_and_does_not_start_web_server(
    tmp_path, monkeypatch, capsys
) -> None:
    started = []
    monkeypatch.setattr(cli, "run_web_server", lambda **kwargs: started.append(kwargs))

    def fail_download(_cache_dir: Path) -> dict[str, object]:
        raise CatalogDownloadError("https://fixed.example/private/cache")

    monkeypatch.setattr(cli, "download_full_catalog", fail_download)

    assert main(["sky-chart", "--download-catalog", "--catalog-cache-dir", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "downloaded": False,
        "error": "catalog_download_failed",
    }
    assert "private" not in captured.err
    assert started == []


def test_download_catalog_unusable_cache_directory_is_stable(
    tmp_path, monkeypatch, capsys
) -> None:
    cache_parent = tmp_path / "not-a-directory"
    cache_parent.write_text("blocked", encoding="utf-8")
    started = []
    monkeypatch.setattr(cli, "run_web_server", lambda **kwargs: started.append(kwargs))

    assert (
        main(
            [
                "sky-chart",
                "--download-catalog",
                "--catalog-cache-dir",
                str(cache_parent / "child"),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "downloaded": False,
        "error": "catalog_download_failed",
    }
    assert str(cache_parent) not in captured.err
    assert started == []


def test_sky_chart_passes_only_loopback_port_open_flag_and_catalog_cache(
    tmp_path: Path, monkeypatch
) -> None:
    observed = []
    monkeypatch.setattr(cli, "run_web_server", lambda **kwargs: observed.append(kwargs))

    assert (
        main(
            [
                "sky-chart",
                "--port",
                "8123",
                "--open",
                "--catalog-cache-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert observed == [
        {"port": 8123, "open_browser": True, "catalog_cache_dir": tmp_path}
    ]


def test_sky_chart_server_start_failure_is_stable(monkeypatch, capsys) -> None:
    def fail_server(**_kwargs: object) -> None:
        raise OSError("bind failed at /private/socket")

    monkeypatch.setattr(cli, "run_web_server", fail_server)

    assert main(["sky-chart"]) == 1

    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "started": False,
        "error": "web_server_start_failed",
    }
    assert "private" not in captured.err


def test_sky_chart_system_exit_failure_is_stable(monkeypatch, capsys) -> None:
    def exit_with_bind_failure(**_kwargs: object) -> None:
        print("uvicorn bind failure at /private/socket", file=sys.stderr)
        raise SystemExit(1)

    monkeypatch.setattr(cli, "run_web_server", exit_with_bind_failure)

    assert main(["sky-chart"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "started": False,
        "error": "web_server_start_failed",
    }


def test_sky_chart_successful_system_exit_is_not_normalized(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_web_server",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit(0)),
    )

    with pytest.raises(SystemExit) as exit_result:
        main(["sky-chart"])

    assert exit_result.value.code == 0
