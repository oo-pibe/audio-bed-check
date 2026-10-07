"""The skill and references must track the code: every flag, every message, every manifest field."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "audio-bed-check"
SRC = ROOT / "src" / "audio_bed_check"

# every sentence a user can see, as fragments; each must still be in src/ and be explained in
# references/cli.md
MESSAGES = [
    "too short to test for a repeat longer than", "not flagged", "too short to measure level steps",
    "nothing to correlate", "min_period must be positive", "threshold must be between", "edge must be >= 0",
    "no speech found in the voiceover above", "the voiceover has no gap of", "s or more between speech runs",
    "window(s) fall outside the mix", "speech windows fall outside the mix",
    "hop must be at least one sample", "must be a finite number, got", "must be a positive number, got",
    "must be zero or more, got", "must be between 0 (exclusive) and 1, got",
    "mix true peak above -1 dBTP", "ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG",
    "no such file", "ffmpeg could not decode it", "it has no audio stream", "ffmpeg not found at",
    "decoded to no audio", "not a WAV stream", "resample to 48 kHz", "no level change anywhere",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must start with frontmatter"
    return dict(line.split(":", 1) for line in m.group(1).splitlines())


def test_skill_frontmatter():
    fm = {k.strip(): v.strip().strip('"') for k, v in frontmatter(read(SKILL / "SKILL.md")).items()}
    assert sorted(fm) == ["description", "license", "name"]
    assert fm["name"] == "audio-bed-check"
    assert 100 < len(fm["description"]) <= 1024
    assert len(read(SKILL / "SKILL.md").splitlines()) <= 250


def _flags(parser):
    for action in parser._actions:
        yield from (s for s in action.option_strings if s.startswith("--"))
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _flags(sub)


def test_every_flag_is_documented():
    from audio_bed_check.cli import build_parser

    flags = set(_flags(build_parser()))
    assert len(flags) >= 10
    reference = read(SKILL / "references" / "cli.md")
    for flag in flags:
        assert re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", reference), (
            f"references/cli.md does not mention {flag}"
        )


def test_every_message_is_in_source_and_explained():
    source = "\n".join(read(p) for p in SRC.glob("*.py"))
    reference = read(SKILL / "references" / "cli.md")
    for message in MESSAGES:
        assert message in source, f'"{message}" is no longer in src/: update MESSAGES and references/cli.md'
        assert message in reference, f'references/cli.md does not explain "{message}"'


def test_every_raised_message_has_a_fragment():
    pattern = re.compile(r'(?:Error|ArgumentTypeError)\(\s*f?"([^"{]+)')
    for path in SRC.glob("*.py"):
        for literal in pattern.findall(read(path)):
            assert any(fragment in literal for fragment in MESSAGES), (
                f'{path.name}: no MESSAGES fragment covers "{literal}"'
            )


def test_relative_links_resolve():
    for path in [ROOT / "README.md", SKILL / "SKILL.md", *(SKILL / "references").glob("*.md")]:
        for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)#\s]+)", read(path)):
            assert (path.parent / target).exists(), f"{path.name} links to missing {target}"


def test_plugin_manifests_agree_with_pyproject():
    version = re.search(r'__version__ = "([^"]+)"', read(SRC / "__init__.py")).group(1)
    plugin = json.loads(read(ROOT / ".claude-plugin" / "plugin.json"))
    market = json.loads(read(ROOT / ".claude-plugin" / "marketplace.json"))
    assert plugin["name"] == "audio-bed-check"
    assert plugin["version"] == version, "bump .claude-plugin/plugin.json with __version__"
    assert market["name"] == "audio-bed-check"
    assert [(p["name"], p["source"]) for p in market["plugins"]] == [("audio-bed-check", "./")]
