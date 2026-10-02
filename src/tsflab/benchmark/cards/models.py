"""Render and validate the canonical, evidence-preserving model-card body."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

from tsflab.tsf_core.paths import repository_root, require_checkout

from tsflab.benchmark.cards.metadata import declared_model_fields, read_model_card


ROOT = repository_root()
START = "<!-- model-card:canonical:start -->"
END = "<!-- model-card:canonical:end -->"
REQUIRED_SECTIONS = (
    "Input and output",
    "Paper and code",
    "Local implementation",
    "Differences",
    "Shared components",
    "Configuration constraints",
)


def _section(text: str, *names: str) -> str:
    """Return an existing level-two section without altering its contents."""
    wanted = {name.casefold() for name in names}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip().casefold() in wanted:
            end = next(
                (
                    cursor
                    for cursor in range(index + 1, len(lines))
                    if lines[cursor].startswith("## ")
                ),
                len(lines),
            )
            return "\n".join(lines[index + 1 : end]).strip()
    return ""


def _link(label: str, url: object) -> str:
    value = str(url or "").strip()
    return f"[{label}]({value})" if value else f"{label}: not available"


def render_canonical_body(card_path: Path) -> str:
    """Project metadata, runtime facts, and preserved notes into standard sections."""
    metadata = read_model_card(card_path)
    spec_path = card_path.with_name("spec.py")
    runtime = declared_model_fields(spec_path)
    config_path = ROOT / str(runtime["config_path"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    params = config.get("model", {}).get("params", {})
    task = runtime.get("contract_task", {})
    capabilities = set(runtime.get("capabilities", ()))
    components = tuple(runtime.get("components", ()))
    paper = metadata["paper"]
    codebase = metadata["codebase"]
    assert isinstance(paper, dict)

    original = card_path.read_text(encoding="utf-8")
    differences = _section(original, "Source and verification", "Verification")
    if not differences:
        differences = (
            "No additional implementation differences are recorded in the preserved "
            "card notes. This is an explicit documentation gap, not an equivalence claim."
        )

    seq_len = task.get("seq_len", "configured") if isinstance(task, dict) else "configured"
    pred_len = task.get("pred_len", "configured") if isinstance(task, dict) else "configured"
    axis = "nodes" if "spatiotemporal" in capabilities else "channels"
    output = f"`[batch, {pred_len}, {axis}]` point forecast"
    if "quantile-output" in capabilities:
        output = f"`[batch, {pred_len}, {axis}, quantiles]` quantile forecast"
    elif "distribution-output" in capabilities:
        output = f"`[batch, {pred_len}, {axis}, parameters]` distribution parameters"
    extra_input = ""
    if "covariate" in capabilities:
        extra_input = " Timestamp or exogenous marks are supplied through the runtime batch contract."
    elif "spatiotemporal" in capabilities:
        extra_input = (
            " Adjacency and temporal/node covariates are supplied only when the "
            "model's executable contract requires them."
        )

    if components:
        component_lines = "\n".join(
            f"- [`{name}`](../_components/{name}/README.md)" for name in components
        )
    else:
        component_lines = (
            "No cataloged shared component is imported; the architecture remains "
            "model-local."
        )
    if params:
        parameter_text = ", ".join(f"`{key}={value!r}`" for key, value in params.items())
    else:
        parameter_text = "No model-specific parameters are set by the default preset."

    paper_bits = [
        _link("paper", paper.get("url")),
        f"title: {paper.get('title') or 'not available'}",
        f"venue/year: {paper.get('venue') or 'not available'} / {paper.get('year') or 'not available'}",
    ]
    code_bits = (
        ["codebase: not available"]
        if codebase is None
        else [
            _link("codebase", codebase.get("url")),
            f"revision: `{codebase.get('revision') or 'not available'}`",
            f"license: `{codebase.get('license') or 'not available'}`",
        ]
    )
    return f"""{START}
## Input and output

The primary input is a history tensor shaped `[batch, {seq_len}, {axis}]`. The
declared output contract is a {output}.{extra_input}

## Paper and code

- {'; '.join(paper_bits)}
- {'; '.join(code_bits)}

## Local implementation

TSFLab implements the model locally after checking the paper and, when
available, the pinned official codebase. Construction and runtime schema live
in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py) (imported, strictly shared building blocks are listed
under Shared components), and the default preset is
[`{runtime['config_path']}`](../../../../{runtime['config_path']}).

## Differences

{differences}

## Shared components

{component_lines}

## Configuration constraints

The contract fixture uses `seq_len={seq_len}` and `pred_len={pred_len}`. Default
model parameters are: {parameter_text}
{END}"""


def update_model_card(card_path: Path) -> bool:
    """Insert or replace the canonical projection while preserving legacy notes."""
    text = card_path.read_text(encoding="utf-8")
    block = render_canonical_body(card_path)
    if START in text:
        updated = re.sub(
            rf"{re.escape(START)}.*?{re.escape(END)}",
            block,
            text,
            flags=re.DOTALL,
        )
    else:
        heading = re.search(r"(?m)^## ", text)
        if heading is None:
            updated = text.rstrip() + "\n\n" + block + "\n"
        else:
            updated = text[: heading.start()].rstrip() + "\n\n" + block + "\n\n" + text[heading.start() :]
    if updated == text:
        return False
    card_path.write_text(updated, encoding="utf-8")
    return True


def audit_model_card_body(card_path: Path) -> list[str]:
    """Check structure and non-empty content of the canonical body projection."""
    text = card_path.read_text(encoding="utf-8")
    problems: list[str] = []
    if text.count(START) != 1 or text.count(END) != 1:
        return [f"{card_path.relative_to(ROOT)} has no unique canonical body block"]
    rendered = render_canonical_body(card_path)
    actual = START + text.split(START, 1)[1].split(END, 1)[0] + END
    if actual != rendered:
        problems.append(
            f"{card_path.relative_to(ROOT)} canonical body is stale; "
            "run `tsf repo cards`"
        )
    block = actual.removeprefix(START).removesuffix(END)
    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        marker = f"## {section}"
        if block.count(marker) != 1:
            problems.append(
                f"{card_path.relative_to(ROOT)} requires one {marker!r} section"
            )
            continue
        positions.append(block.index(marker))
        if not _section(block, section):
            problems.append(f"{card_path.relative_to(ROOT)} has empty {marker!r}")
    if positions != sorted(positions):
        problems.append(f"{card_path.relative_to(ROOT)} canonical sections are out of order")
    return problems


def documentation_gaps(card_path: Path) -> list[str]:
    """Report evidence gaps without converting them into invented claims."""
    original = card_path.read_text(encoding="utf-8")
    outside = re.sub(
        rf"{re.escape(START)}.*?{re.escape(END)}", "", original, flags=re.DOTALL
    )
    gaps: list[str] = []
    if not _section(outside, "Source and verification", "Verification"):
        gaps.append("differences")
    metadata = read_model_card(card_path)
    summary = str(metadata["summary"])
    if len(summary) < 80:
        gaps.append("core-architecture")
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        require_checkout("model-card regeneration")
    cards = sorted(
        card
        for card in (ROOT / "src" / "tsflab" / "models").glob("*/README.md")
        if not card.parent.name.startswith("_")
    )
    changed = sum(update_model_card(card) for card in cards) if args.write else 0
    problems = [problem for card in cards for problem in audit_model_card_body(card)]
    gaps = {str(card.relative_to(ROOT)): documentation_gaps(card) for card in cards}
    gaps = {path: values for path, values in gaps.items() if values}
    print(f"model cards: {len(cards)}; changed: {changed}; structural failures: {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")
    print(f"cards with evidence gaps: {len(gaps)}")
    for path, values in gaps.items():
        print(f"  - {path}: {', '.join(values)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
