"""Static tests of the 11-command CLI surface, argument routing, and `tsf init` modules."""

from __future__ import annotations

import tomllib

import pytest

from tsflab.cli import main as cli
from tsflab.cli.commands.catalog_resources import _extract_kind
from tsflab.cli.commands.data_results import _extract_from
from tsflab.cli.commands.execution import _extract_backend
from tsflab.agent.scaffold import extras_for, init_project, render_agents_md, sync_command
from tsflab.agent.modules import CHAIN, find_project, module_skills, parse_modules

COMMANDS = {
    "catalog", "data", "model", "run", "env", "result", "realtime",
    "research", "repo", "agent", "init",
}


def test_public_surface_is_eleven_module_commands() -> None:
    assert set(cli.COMMANDS) == COMMANDS
    assert all(f"    {name} " in cli.__doc__ for name in COMMANDS)


@pytest.mark.parametrize("old", [
    "dataset", "component", "verify", "smoke", "inspect", "queue", "slurm", "storage",
    "usage", "interface", "submit", "hub", "schema-export", "leaderboard-build",
])
def test_removed_commands_are_unknown(old: str, capsys) -> None:
    assert cli.main([old]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["repo", "audit"], ["repo", "doctor"], ["model", "show", "x"],
                                  ["model", "list"], ["data", "list"], ["result", "nope"]])
def test_removed_subcommands_print_usage(argv: list[str], capsys) -> None:
    assert cli.main(argv) == 2
    assert "usage:" in capsys.readouterr().err


def test_help_for_every_routed_command(capsys) -> None:
    assert cli.main([]) == 0
    for name in ("catalog", "data", "model", "repo", "agent", "result"):
        assert cli.main([name, "--help"]) == 0
    assert "tsf catalog show" in capsys.readouterr().out


def test_option_extractors() -> None:
    assert _extract_kind(["show", "x", "--kind", "model", "--json"]) == ("model", ["show", "x", "--json"])
    assert _extract_kind(["--kind=dataset", "a"]) == ("dataset", ["a"])
    assert _extract_from(["--from", "gift", "--link-only"]) == ("gift", ["--link-only"])
    assert _extract_backend(["add", "dir", "--backend", "queue"]) == (["add", "dir"], "queue")
    assert _extract_backend(["cfg.toml"]) == (["cfg.toml"], "local")
    with pytest.raises(SystemExit):
        _extract_backend(["--backend", "ray"])


def test_module_selection_and_extras() -> None:
    assert parse_modules(None) == list(CHAIN)
    assert parse_modules("models,data") == ["data", "models"]
    with pytest.raises(ValueError):
        parse_modules("data,bogus")
    assert extras_for(list(CHAIN)) == ["all"]
    assert extras_for(["data", "release"]) == ["data", "hub", "realtime"]
    assert "audit" in module_skills(["models"], {"intake": ["audit"]})


def test_maintenance_is_opt_in() -> None:
    assert "maintenance" not in parse_modules(None)
    assert parse_modules("maintenance,data") == ["data", "maintenance"]
    assert extras_for([*CHAIN, "maintenance"]) == ["all"]
    assert extras_for(["maintenance"]) == []


def test_generated_agents_md_fits_the_context_budget() -> None:
    assert len(render_agents_md("p", list(CHAIN)).splitlines()) <= 50
    assert len(render_agents_md("p", [*CHAIN, "maintenance"]).splitlines()) <= 50


def test_init_scaffolds_selected_modules(tmp_path, monkeypatch) -> None:
    project = tmp_path / "proj"
    init_project(project, modules=["data", "autoresearch"])
    table = tomllib.loads((project / "pyproject.toml").read_text())
    assert table["tool"]["tsflab"]["modules"] == ["data", "autoresearch"]
    assert table["project"]["dependencies"] == ["tsflab[data,autoresearch]"]
    skills = {p.name for p in (project / ".agents" / "skills").iterdir()}
    assert skills == {"add-dataset", "inspect-dataset", "run-autoresearch"}
    assert (project / ".agents" / "tasks" / "autoresearch.toml").is_file()
    index = (project / ".agents" / "README.md").read_text()
    assert "run-autoresearch" in index and "add-model" not in index and "audit" not in index
    assert not (project / ".agents" / "tasks" / "experiment.toml").exists()
    agents = (project / "AGENTS.md").read_text()
    assert "tsf catalog" in agents and "run-autoresearch" in agents and "add-model" not in agents
    assert (project / "CLAUDE.md").read_text() == agents
    assert "tsflab://" in (project / "configs/runs/example.toml").read_text()
    assert find_project(project / "configs") == (project.resolve(), ["data", "autoresearch"])
    with pytest.raises(FileExistsError):
        init_project(project)

    monkeypatch.chdir(project)
    (project / "AGENTS.md").write_text(agents + "my note\n")
    assert sync_command(["--add", "models"]) == 0
    refreshed = (project / "AGENTS.md").read_text()
    assert "my note" in refreshed and "add-model" in refreshed
    assert (project / ".agents" / "tasks" / "intake.toml").is_file()
    assert tomllib.loads((project / "pyproject.toml").read_text())["tool"]["tsflab"]["modules"] == [
        "data", "models", "autoresearch"]
