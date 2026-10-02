"""Progressive-disclosure views of model, component, and dataset cards.

One depth model serves every catalog resource, so an agent can spend context
only as needed:

* L0 - one line: ``name``, ``kind``, ``summary``, ``tags``.
* L1 - front matter plus the interface/constraint sections of the card.
* L2 - the full card.
* L3 - the source, config, test, and evidence paths to open next.

The reader is generic: it needs only a card's flat front matter and its
level-two sections, so card content can evolve without touching this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from moderntsf.benchmark.catalog_metadata import read_front_matter

DEPTHS = (0, 1, 2, 3)
DEPTH_HELP = (
    "0 one-line summary, 1 front matter + interface/constraints, "
    "2 full card, 3 paths to open"
)
L0_SUMMARY_CHARS = 160
# Sections shown at L1: how the resource is called and what limits composition.
_L1_SECTION = re.compile(
    r"input|output|interface|contract|constraint|when to use", re.IGNORECASE
)


@dataclass(frozen=True)
class Card:
    """A parsed card: flat front matter plus ordered level-two sections."""

    path: Path
    front: dict[str, object]
    text: str
    sections: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return str(self.front.get("name", self.path.parent.name))

    @property
    def kind(self) -> str:
        return str(self.front.get("kind") or "model")

    @property
    def summary(self) -> str:
        return " ".join(str(self.front.get("summary", "")).split())

    @property
    def tags(self) -> tuple[str, ...]:
        return card_tags(self.front)


def split_sections(text: str) -> tuple[tuple[str, str], ...]:
    """Split a card body on level-two headings, ignoring fenced code blocks."""
    sections: list[tuple[str, list[str]]] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and re.fullmatch(r"<!-- [\w:-]+ -->", line.strip()):
            continue  # generated-block markers are not content
        if not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return tuple((title, "\n".join(body).strip()) for title, body in sections)


def read_card(path: Path) -> Card:
    """Parse one README card; raise ``ValueError`` without front matter."""
    front = read_front_matter(path)
    text = path.read_text(encoding="utf-8")
    return Card(path=path, front=front, text=text, sections=split_sections(text))


def card_tags(front: dict[str, object]) -> tuple[str, ...]:
    """Return explicit tags, else derive them from list fields and the loader."""
    tags = front.get("tags")
    if isinstance(tags, list):
        return tuple(str(tag) for tag in tags)
    derived: list[str] = []
    for key, value in front.items():
        if key in {"tags", "origin_models"}:
            continue
        if isinstance(value, list):
            derived.extend(str(item) for item in value)
    for key in ("category", "loader"):
        if front.get(key):
            derived.append(str(front[key]))
    return tuple(dict.fromkeys(derived))


def truncate_summary(summary: str, limit: int = L0_SUMMARY_CHARS) -> str:
    """Return the first sentence, shortened to ``limit`` characters."""
    summary = " ".join(summary.split())
    first = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", summary, maxsplit=1)[0]
    if len(first) <= limit:
        return first
    return first[: limit - 3].rstrip(" ,;:") + "..."


def l0_record(
    name: str, kind: str, summary: str, tags: tuple[str, ...] | list[str]
) -> dict[str, object]:
    """Return the structured L0 record for one resource."""
    return {
        "name": name,
        "kind": kind,
        "summary": truncate_summary(summary),
        "tags": list(tags),
    }


def l0_line(record: dict[str, object]) -> str:
    """Format an L0 record as one tab-separated line: name, kind, summary, tags."""
    tags = ",".join(str(tag) for tag in record["tags"])  # type: ignore[union-attr]
    return f"{record['name']}\t{record['kind']}\t{record['summary']}\t{tags}"


def card_l0(card: Card) -> dict[str, object]:
    return l0_record(card.name, card.kind, card.summary, card.tags)


def front_matter_text(card: Card) -> str:
    """Return the card's front matter block verbatim, delimiters included."""
    lines = card.text.splitlines()
    end = lines.index("---", 1)
    return "\n".join(lines[: end + 1])


def l1_sections(card: Card) -> tuple[tuple[str, str], ...]:
    """Return the interface/constraint sections shown at L1."""
    return tuple(
        (title, body) for title, body in card.sections if _L1_SECTION.search(title)
    )


def render_text(
    card: Card,
    depth: int,
    *,
    facts: dict[str, object] | None = None,
    paths: list[str] | None = None,
) -> str:
    """Render ``card`` at ``depth`` as text.

    ``facts`` are runtime facts (for example config or smoke-config paths) that
    live outside the card; they are listed after the front matter at L1.
    ``paths`` are the files to open at L3.
    """
    line = l0_line(card_l0(card))
    if depth == 0:
        return line
    if depth == 3:
        return "\n".join([line, "", *(f"- {path}" for path in (paths or []))])
    if depth == 2:
        return card.text.rstrip() + "\n"
    parts = [front_matter_text(card)]
    if facts:
        parts.append(
            "Runtime facts:\n"
            + "\n".join(f"- {key}: {_format_fact(value)}" for key, value in facts.items())
        )
    for title, body in l1_sections(card):
        parts.append(f"## {title}\n\n{body}")
    return "\n\n".join(parts) + "\n"


def _format_fact(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    return str(value)


def card_payload(
    card: Card,
    depth: int,
    *,
    facts: dict[str, object] | None = None,
    paths: list[str] | None = None,
) -> dict[str, object]:
    """Return the structured form of ``render_text`` for ``--json`` output."""
    payload: dict[str, object] = {"depth": depth, **card_l0(card)}
    if depth in (1, 2):
        payload["front_matter"] = card.front
        payload["facts"] = facts or {}
    if depth == 1:
        payload["sections"] = dict(l1_sections(card))
    if depth == 2:
        payload["text"] = card.text
    if depth == 3:
        payload["paths"] = paths or []
    return payload
