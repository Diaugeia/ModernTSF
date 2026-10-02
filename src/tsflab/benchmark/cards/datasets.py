"""Dataset cards: curated descriptive facts plus a generated runtime block.

A dataset card mirrors a model card. Everything a person had to research
(domain, source, license, statistics, protocol, pitfalls) is curated prose and
front matter that this module never rewrites. Everything derivable from the
executable preset (loader, config, task modes, parameters, files) lives in one
marked block and in a few front-matter keys that are regenerated.

Front matter is flat, one ``key: <JSON value>`` pair per line, so cards stay
greppable and an agent can read level 0/1 facts without the body. Cards for a
family of series sharing a loader (GIFT-Eval) get one family card as well.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tomllib

START = "<!-- dataset-card:canonical:start -->"
END = "<!-- dataset-card:canonical:end -->"

#: Generated keys; the generator owns them and the audit fails when they drift.
RUNTIME_KEYS = ("name", "kind", "config", "loader", "alias", "task_modes")
#: Curated keys in canonical order. Everything here is descriptive truth.
CURATED_KEYS = (
    "summary",
    "domain",
    "tags",
    "source",
    "source_url",
    "citation",
    "citation_url",
    "license",
    "redistribution",
    "frequency",
    "time_span",
    "length",
    "channels",
    "channel_kind",
    "target",
    "missing_values",
    "protocol",
    "literature_protocol",
    "seq_lens",
    "pred_lens",
    "split",
    "stats_basis",
    "related",
    "realtime_track",
)
#: Keys every preset card must carry; ``OPTIONAL_KEYS`` may be omitted.
#: ``protocol`` is the TSFLab protocol; ``literature_protocol`` is context only.
OPTIONAL_KEYS = ("realtime_track", "literature_protocol")
REQUIRED_PRESET_KEYS = tuple(key for key in CURATED_KEYS if key not in OPTIONAL_KEYS)
#: A family card describes a collection, so per-series facts are not required.
REQUIRED_FAMILY_KEYS = (
    "summary", "domain", "tags", "source", "source_url", "citation", "citation_url",
    "license", "redistribution", "frequency", "protocol", "stats_basis",
)
REDISTRIBUTION = ("allowed", "conditional", "restricted", "unknown")
#: Top-level domains; a card's ``domain`` is ``"<Top>"`` or ``"<Top> / <detail>"``.
DOMAINS = ("Energy", "Transport", "Environment", "Weather", "Finance", "Healthcare",
           "Nature", "Sales", "Web/CloudOps", "General")
STATS_BASIS = ("measured", "source-reported", "mixed")
CHANNEL_KINDS = ("channels", "nodes", "series", "stations")
CURATED_SECTIONS = (
    "Overview",
    "Provenance and license",
    "Structure and statistics",
    "Standard protocol and known pitfalls",
)
TRAILING_SECTIONS = ("Related datasets",)
GENERATED_SECTIONS = (
    "Loader and files",
    "Input and output contract",
    "Dataset parameters",
    "Task overrides",
    "Preparation and use",
    "Composition constraints",
)
FAMILY_GENERATED_SECTIONS = ("Members and loader", "Composition constraints")
TODO = "TODO"


@dataclass(frozen=True)
class DatasetRecord:
    """Resolved metadata for one runnable dataset configuration preset."""

    name: str
    config: str
    loader: str
    alias: str
    path: str
    dataset_id: str
    params: dict[str, object]
    task: dict[str, object]
    track: str
    task_modes: tuple[str, ...]


def _quoted(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _task_modes_for_loader(loader: str) -> tuple[str, ...]:
    """Resolve modes from the executable registry instead of duplicating them."""
    from tsflab.benchmark.registry.datasets import (
        DATASET_REGISTRY,
        ordered_task_modes,
        register_dataset_by_name,
    )

    register_dataset_by_name(loader)
    return ordered_task_modes(DATASET_REGISTRY.get(loader).task_modes)


def dataset_records(root: Path) -> tuple[DatasetRecord, ...]:
    """Read every dataset preset without importing its runtime dependencies."""
    config_root = root / "configs" / "datasets"
    records: list[DatasetRecord] = []
    for path in sorted(config_root.rglob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset", {})
        relative = path.relative_to(config_root)
        name = relative.with_suffix("").as_posix()
        records.append(
            DatasetRecord(
                name=name,
                config=path.relative_to(root).as_posix(),
                loader=str(dataset.get("name", "")),
                alias=str(dataset.get("alias", name)),
                path=str(dataset.get("path", "")),
                dataset_id=str(dataset.get("id", "")),
                params=dict(dataset.get("params", {})),
                task=dict(payload.get("task", {})),
                track=str(dataset.get("track", "")),
                task_modes=_task_modes_for_loader(str(dataset.get("name", ""))),
            )
        )
    return tuple(records)


def dataset_card_path(root: Path, name: str) -> Path:
    """Return the canonical dataset-card path."""
    return root / "catalog" / "datasets" / Path(name) / "README.md"


def family_names(records: tuple[DatasetRecord, ...]) -> tuple[str, ...]:
    """Loaders whose presets select a series by ``id`` own one family card."""
    return tuple(sorted({record.loader for record in records if record.dataset_id}))


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

def split_card(text: str) -> tuple[dict[str, object], str]:
    """Split a card into ordered front-matter values and its body.

    Values are JSON, so lists and numbers stay typed without a YAML dependency.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("card has no front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("card front matter is unterminated") from exc
    front: dict[str, object] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, raw = line.partition(":")
        if not separator or key != key.strip() or not key:
            raise ValueError(f"invalid front-matter line: {line!r}")
        try:
            front[key] = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"front-matter value of {key!r} is not JSON") from exc
    return front, "\n".join(lines[end + 1 :]).strip("\n")


def _dump(front: dict[str, object]) -> str:
    return "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in front.items())


def read_dataset_facts(path: Path) -> dict[str, object]:
    """Return the level-0/1 front-matter facts of one card."""
    return split_card(path.read_text(encoding="utf-8"))[0]


def _placeholder(key: str) -> object:
    return [] if key in {"tags", "seq_lens", "pred_lens", "related"} else TODO


def _front_matter(
    runtime: dict[str, object], existing: dict[str, object], required: tuple[str, ...]
) -> dict[str, object]:
    """Merge curated values (kept verbatim) with regenerated runtime keys."""
    front: dict[str, object] = {"name": runtime["name"], "kind": runtime["kind"]}
    for key in CURATED_KEYS:
        if key in existing:
            front[key] = existing[key]
        elif key in required:
            front[key] = _placeholder(key)
    for key, value in existing.items():
        if key not in front and key not in RUNTIME_KEYS:
            front[key] = value  # unknown keys survive; the audit reports them
    for key in RUNTIME_KEYS[2:]:
        if key in runtime:
            front[key] = runtime[key]
    return front


# ---------------------------------------------------------------------------
# Generated blocks
# ---------------------------------------------------------------------------

def _item_contract(record: DatasetRecord) -> str:
    if record.loader == "gift_eval":
        return (
            "Windowed history/target values and timestamp marks; after batching, "
            "values use `[batch, time, channels]`."
        )
    if record.loader in {"cauair_st", "synthetic_st", "ultratraffic_st", "realtime_panel_st"}:
        return (
            "Each item is `(value_history, value_future, covariate_history, covariate_future)`; "
            "values use `[time, nodes]` and covariates `[time, nodes, features]` before batching."
        )
    if record.loader in {"cauair_ts", "ultratraffic_ts", "realtime_panel_ts"}:
        return (
            "Each item contains history/future values shaped `[time, nodes]` (nodes become "
            "channels) plus timestamp marks before batching; covariates are dropped."
        )
    return (
        "Each item provides history/target windows and timestamp marks; after batching, "
        "values use `[batch, time, channels]`."
    )


def render_runtime_block(record: DatasetRecord) -> str:
    """Render the generated block from the executable preset only."""
    modes = ", ".join(f"`{mode}`" for mode in record.task_modes)
    params = json.dumps(record.params, ensure_ascii=False, indent=2, sort_keys=True)
    task = json.dumps(record.task, ensure_ascii=False, indent=2, sort_keys=True)
    track = record.track or "standard"
    root_prefix = "../" * (3 + record.name.count("/"))
    return f"""{START}
## Loader and files

- Registry loader: `{record.loader}`
- Config: [`{record.config}`]({root_prefix}{record.config})
- Local path: `{record.path or '(loader-defined)'}`
- Dataset id: `{record.dataset_id or '(not applicable)'}`
- Track: `{track}`

## Input and output contract

{_item_contract(record)}

Sequence length, label length, feature mode, and batch size are supplied by the
experiment task unless explicitly overridden below.

## Dataset parameters

```json
{params}
```

## Task overrides

```json
{task}
```

## Preparation and use

Inspect availability with `tsf data inspect --config {record.config}`; fetch
published files with `tsf data download {record.name}` when the preset is
listed by `tsf data download --list`, otherwise place the data at the local
path above (see `tsf data prepare --help`). Reference this preset from an
experiment configuration rather than duplicating its loader parameters.

## Composition constraints

Task modes: {modes}. Match one of them to the model's declared task mode; the
config loader rejects other combinations. Change feature or scaling parameters
only after reading the loader. Paths are repository defaults and may need local
overrides; the card does not imply that the data is bundled.
{END}"""


def render_family_block(loader: str, members: tuple[DatasetRecord, ...], facts: dict[str, dict[str, object]]) -> str:
    """Render the member table from the child cards (read-only projection)."""
    rows = ["| Preset | Domain | Frequency | Horizon |", "| --- | --- | --- | --- |"]
    for record in members:
        child = facts.get(record.name, {})
        link = f"[`{record.name}`]({record.name.split('/', 1)[1]}/README.md)"
        horizon = record.task.get("pred_len", "")
        rows.append(f"| {link} | {child.get('domain', '')} | {child.get('frequency', '')} | {horizon} |")
    modes = ", ".join(f"`{mode}`" for mode in members[0].task_modes)
    return f"""{START}
## Members and loader

- Registry loader: `{loader}`
- Data root: `{members[0].path}`; each preset selects a series by `id`.
- Presets: {len(members)}

{chr(10).join(rows)}

## Composition constraints

Task modes: {modes}. Each preset fixes `pred_len` and `features`; change them
only through an explicit task override. The family shares one loader, so
loader-level facts (splitting, scaling) are documented in the per-series
generated blocks.
{END}"""


# ---------------------------------------------------------------------------
# Rendering and writing
# ---------------------------------------------------------------------------

def _skeleton(title: str, sections: tuple[str, ...]) -> str:
    body = "\n\n".join(f"## {section}\n\n{TODO}" for section in sections)
    return f"# {title}\n\n{body}"


def _assemble(front: dict[str, object], existing_body: str, block: str, title: str,
              curated: tuple[str, ...], trailing: tuple[str, ...]) -> str:
    if START in existing_body and END in existing_body:
        pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
        body = pattern.sub(lambda _match: block, existing_body, count=1)
    elif existing_body.strip():
        marker = re.search(r"(?m)^## " + re.escape(trailing[0]) + r"\s*$", existing_body) if trailing else None
        if marker:
            body = existing_body[: marker.start()].rstrip() + "\n\n" + block + "\n\n" + existing_body[marker.start():]
        else:
            body = existing_body.rstrip() + "\n\n" + block
    else:
        head = _skeleton(title, curated)
        tail = _skeleton(title, trailing).split("\n\n", 1)[1] if trailing else ""
        body = head + "\n\n" + block + ("\n\n" + tail if tail else "")
    return f"---\n{_dump(front)}\n---\n\n{body.strip(chr(10))}\n"


def render_dataset_card(record: DatasetRecord, existing: str | None = None) -> str:
    """Render a preset card, preserving every curated byte of ``existing``."""
    front_in, body_in = split_card(existing) if existing else ({}, "")
    runtime = {
        "name": record.name, "kind": "dataset", "config": record.config,
        "loader": record.loader, "alias": record.alias, "task_modes": list(record.task_modes),
    }
    front = _front_matter(runtime, front_in, REQUIRED_PRESET_KEYS)
    return _assemble(front, body_in, render_runtime_block(record), record.alias,
                     CURATED_SECTIONS, TRAILING_SECTIONS)


def render_family_card(loader: str, members: tuple[DatasetRecord, ...], root: Path,
                       existing: str | None = None) -> str:
    """Render a family card; its member table is projected from child cards."""
    front_in, body_in = split_card(existing) if existing else ({}, "")
    facts = {}
    for record in members:
        path = dataset_card_path(root, record.name)
        if path.is_file():
            try:
                facts[record.name] = read_dataset_facts(path)
            except ValueError:
                pass
    runtime = {"name": loader, "kind": "dataset-family"}
    front = _front_matter(runtime, front_in, REQUIRED_FAMILY_KEYS)
    return _assemble(front, body_in, render_family_block(loader, members, facts), loader,
                     CURATED_SECTIONS, TRAILING_SECTIONS)


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def expected_dataset_cards(root: Path) -> dict[Path, str]:
    """Return each dataset card as it should be, given its current curated content."""
    records = dataset_records(root)
    expected = {
        dataset_card_path(root, record.name): render_dataset_card(
            record, _read(dataset_card_path(root, record.name))
        )
        for record in records
    }
    for loader in family_names(records):
        members = tuple(record for record in records if record.loader == loader and record.dataset_id)
        path = dataset_card_path(root, loader)
        expected[path] = render_family_card(loader, members, root, _read(path))
    return expected


def write_dataset_cards(root: Path) -> int:
    """Regenerate runtime parts of every dataset card and return the card count."""
    expected = expected_dataset_cards(root)
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if _read(path) != content:
            path.write_text(content, encoding="utf-8")
    return len(expected)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _section_text(body: str, name: str) -> str | None:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"## {name}":
            stop = next((i for i in range(index + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            return "\n".join(lines[index + 1 : stop]).strip()
    return None


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and TODO not in value


def _is_url(value: object) -> bool:
    return value == "n/a" or (isinstance(value, str) and re.match(r"https?://\S+$", value) is not None)


def _check_facts(label: str, front: dict[str, object], required: tuple[str, ...], family: bool) -> list[str]:
    errors = []
    allowed = set(RUNTIME_KEYS) | set(CURATED_KEYS)
    for key in front:
        if key not in allowed:
            errors.append(f"{label}: unknown front-matter field {key!r}")
    for key in required:
        if key not in front:
            errors.append(f"{label}: missing front-matter field {key!r}")
    text_keys = ("summary", "domain", "source", "citation", "license", "frequency", "protocol")
    if not family:
        text_keys += ("time_span", "target", "missing_values", "split")
    for key in text_keys:
        if key in front and not _is_text(front[key]):
            errors.append(f"{label}: front-matter field {key!r} is empty or a TODO placeholder")
    summary = front.get("summary")
    if _is_text(summary):
        text = str(summary)
        if not 40 <= len(text) <= 320:
            errors.append(f"{label}: summary must be one specific sentence of 40-320 characters")
        if "preset loaded by" in text or "Time-series forecasting preset" in text:
            errors.append(f"{label}: summary is generated boilerplate, not a specific description")
    for key in ("source_url", "citation_url"):
        if key in front and not _is_url(front[key]):
            errors.append(f"{label}: {key} must be an http(s) URL or \"n/a\"")
    tags = front.get("tags")
    if "tags" in front and (not isinstance(tags, list) or len(tags) < 3
                            or not all(_is_text(tag) for tag in tags)):
        errors.append(f"{label}: tags must list at least three retrieval terms")
    if "domain" in front and str(front["domain"]).split(" / ", 1)[0] not in DOMAINS:
        errors.append(f"{label}: domain must start with one of {', '.join(DOMAINS)}")
    if "redistribution" in front and front["redistribution"] not in REDISTRIBUTION:
        errors.append(f"{label}: redistribution must be one of {', '.join(REDISTRIBUTION)}")
    if str(front.get("license", "")).strip().lower() == "unknown" and front.get("redistribution") not in (None, "unknown"):
        errors.append(f"{label}: an unknown license requires redistribution \"unknown\"")
    if "stats_basis" in front and front["stats_basis"] not in STATS_BASIS:
        errors.append(f"{label}: stats_basis must be one of {', '.join(STATS_BASIS)}")
    if family:
        return errors
    if "channel_kind" in front and front["channel_kind"] not in CHANNEL_KINDS:
        errors.append(f"{label}: channel_kind must be one of {', '.join(CHANNEL_KINDS)}")
    for key in ("length", "channels"):
        value = front.get(key)
        if key in front and not (isinstance(value, int) and not isinstance(value, bool) and value > 0
                                 or _is_text(value)):
            errors.append(f"{label}: {key} must be a positive integer or descriptive text")
    for key in ("seq_lens", "pred_lens"):
        value = front.get(key)
        if key in front and not (isinstance(value, list) and all(isinstance(v, int) and v > 0 for v in value)):
            errors.append(f"{label}: {key} must be a list of positive integers")
    if "pred_lens" in front and front["pred_lens"] == []:
        errors.append(f"{label}: pred_lens must name at least one horizon")
    return errors


def audit_dataset_cards(root: Path) -> list[str]:
    """Report missing, stale, incomplete, or placeholder dataset cards."""
    records = dataset_records(root)
    names = {record.name for record in records}
    errors: list[str] = []
    expected = expected_dataset_cards(root)
    families = set(family_names(records))
    for path, content in expected.items():
        rel = path.relative_to(root).as_posix()
        actual = _read(path)
        if actual is None:
            errors.append(f"missing resource card: {rel}")
            continue
        try:
            front, body = split_card(actual)
        except ValueError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        if actual != content:
            errors.append(f"stale resource card: {rel} (run `tsf repo cards`)")
        family = path.parent.name in families and path.parent.parent == root / "catalog" / "datasets"
        required = REQUIRED_FAMILY_KEYS if family else REQUIRED_PRESET_KEYS
        errors.extend(_check_facts(rel, front, required, family))
        for related in front.get("related", []) if isinstance(front.get("related"), list) else []:
            if related not in names:
                errors.append(f"{rel}: related dataset {related!r} is not a preset")
        track = front.get("realtime_track")
        if track is not None and not (root / "configs" / "realtime" / f"{track}.toml").is_file():
            errors.append(f"{rel}: realtime_track {track!r} has no configs/realtime/{track}.toml")
        errors.extend(_check_body(rel, body, front, family))
    return errors


def _check_body(rel: str, body: str, front: dict[str, object], family: bool) -> list[str]:
    errors = []
    if body.count(START) != 1 or body.count(END) != 1:
        return [f"{rel}: needs exactly one generated block"]
    outside = re.sub(re.escape(START) + r".*?" + re.escape(END), "", body, flags=re.DOTALL)
    start, end = body.index(START), body.index(END)
    for section in CURATED_SECTIONS + TRAILING_SECTIONS:
        pattern = re.compile(rf"(?m)^## {re.escape(section)}[ \t]*$")
        hits = list(pattern.finditer(body))
        if len(hits) != 1:
            errors.append(f"{rel}: requires exactly one '## {section}' section")
            continue
        text = _section_text(outside, section)
        if not text or TODO in text:
            errors.append(f"{rel}: section '## {section}' is empty or a TODO placeholder")
        trailing = section in TRAILING_SECTIONS
        if (trailing and hits[0].start() < end) or (not trailing and hits[0].start() > start):
            errors.append(f"{rel}: section '## {section}' is out of order")
    basis = front.get("stats_basis")
    stats = (_section_text(outside, "Structure and statistics") or "").casefold()
    if basis in {"measured", "mixed"} and "measured" not in stats:
        errors.append(f"{rel}: statistics section must label measured values")
    if basis in {"source-reported", "mixed"} and "source-reported" not in stats:
        errors.append(f"{rel}: statistics section must label source-reported values")
    return errors


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

SEARCH_FIELDS = ("summary", "domain", "tags", "frequency", "source", "protocol", "license", "time_span", "target")


def dataset_facts(root: Path, names: list[str] | None = None) -> dict[str, dict[str, object]]:
    """Read level-0/1 facts for presets (and families) that already have a card."""
    wanted = names or [record.name for record in dataset_records(root)]
    facts: dict[str, dict[str, object]] = {}
    for name in wanted:
        path = dataset_card_path(root, name)
        if path.is_file():
            try:
                facts[name] = read_dataset_facts(path)
            except ValueError:
                continue
    return facts


def search_text(record: DatasetRecord, facts: dict[str, object]) -> str:
    """Searchable text: runtime identifiers plus curated retrieval facts."""
    parts = [record.name, record.alias, record.loader, record.dataset_id or record.path, record.track]
    for key in SEARCH_FIELDS:
        value = facts.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).casefold()
