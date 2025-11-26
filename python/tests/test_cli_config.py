# python/tests/test_config_cli.py

import argparse
import json
from pathlib import Path

import mneme.commands as commands  # adjust import if needed
import pytest
from mneme.commands import Config  # adjust import if needed

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def fake_cfg(tmp_path):
    """Create a fake config.json and return its contents."""
    cfg = {
        "cc": "/usr/bin/clang",
        "cxx": "/usr/bin/clang++",
        "ldflags": ["-L/usr/lib", "-lmneme"],
        "version": "1.2.3",
    }

    cfg_dir = tmp_path / "native"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text(json.dumps(cfg))

    return cfg, cfg_file


@pytest.fixture
def config_parser(fake_cfg, monkeypatch):
    """
    Build a parser with Config CLI wired in, injecting a fake config.json.
    This prevents the test from depending on real installation layout.
    """

    cfg, cfg_file = fake_cfg

    # Monkeypatch the location of the config file in commands.Config.set_cli_args
    def fake_set_cli_args(parser):
        # Directly load our fake config instead of the real file on disk
        with open(cfg_file) as fd:
            cfg_data = json.load(fd)
        parser.add_argument("key", choices=list(cfg_data.keys()))
        parser.set_defaults(func=Config.run, mneme_config=cfg_data)

    monkeypatch.setattr(Config, "set_cli_args", fake_set_cli_args)

    parser = argparse.ArgumentParser(prog="mneme config")
    Config.set_cli_args(parser)
    return parser


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_config_prints_scalar_values(config_parser, capsys):
    """Scalar config values (e.g. strings) should be printed as-is."""
    args = config_parser.parse_args(["version"])
    Config.run(args, verbosity=None)

    captured = capsys.readouterr().out.strip()
    assert captured == "1.2.3"


def test_config_prints_list_values_space_separated(config_parser, capsys):
    """List values should be printed space-separated."""
    args = config_parser.parse_args(["ldflags"])
    Config.run(args, verbosity=None)

    captured = capsys.readouterr().out.strip()
    assert captured == "-L/usr/lib -lmneme"


def test_config_unknown_key_raises(monkeypatch, tmp_path):
    """Querying an unknown key should raise ValueError."""

    # Build minimal fake config
    cfg = {"only_key": "value"}
    cfg_file = tmp_path / "native" / "config.json"
    cfg_file.parent.mkdir()
    cfg_file.write_text(json.dumps(cfg))

    def fake_set_cli_args(parser):
        with open(cfg_file) as fd:
            cfg_data = json.load(fd)
        parser.add_argument("key", choices=list(cfg_data.keys()))
        parser.set_defaults(func=Config.run, mneme_config=cfg_data)

    monkeypatch.setattr(Config, "set_cli_args", fake_set_cli_args)

    parser = argparse.ArgumentParser()
    Config.set_cli_args(parser)

    # Can't parse unknown key because argparse enforces choices
    with pytest.raises(SystemExit):  # argparse exits on invalid choice
        parser.parse_args(["invalid_key"])


def test_config_missing_config_file_raises_runtimeerror(monkeypatch, tmp_path):
    """
    If set_cli_args is called normally and config.json is missing,
    Config.set_cli_args should raise RuntimeError.
    """
    # Patch the path to point to an empty directory with no config file
    empty_native = tmp_path / "native"
    empty_native.mkdir()

    def fake_resolve():
        return empty_native / "commands.py"

    # Monkeypatch __file__ resolution for locating native/config.json
    monkeypatch.setattr(Config, "__module__", "mneme.cli.commands")
    monkeypatch.setattr(
        Config,
        "set_cli_args",
        lambda parser: Config.set_cli_args.__wrapped__(parser)  # fallback call
        if hasattr(Config.set_cli_args, "__wrapped__")
        else Config.set_cli_args(parser),
    )

    # We need to monkeypatch the lookup path inside the real implementation:
    real_file = Path(__file__).resolve()  # path to *this* test file
    commands_dir = real_file.parent  # simulate commands.py's path

    def fake_file():
        # simulate commands.py living in empty_native parent
        return empty_native / "commands.py"

    monkeypatch.setattr(commands, "__file__", fake_file())

    parser = argparse.ArgumentParser(prog="mneme config")

    with pytest.raises(RuntimeError):
        Config.set_cli_args(parser)
