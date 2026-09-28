#!/usr/bin/env python3
"""Validate every submission bundle in submissions/ against the TSF-Core contract.

v1 checks (per the agreed scope):
  - submission.json matches pipeline/contract.schema.json
  - the run is bound to ModernTSF: each dataset declares a `source_config`
    (a ModernTSF config path) — the binding handle. (A later version will also
    cross-check model ids against the ModernTSF registry.)

Exit code is non-zero if anything fails, so CI can gate PRs.

Usage:  python3 pipeline/validate.py
"""
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    sys.exit("missing dependency: pip install jsonschema")

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "pipeline" / "contract.schema.json").read_text())
VALIDATOR = Draft7Validator(SCHEMA)


def check(bundle: Path) -> list[str]:
    errs: list[str] = []
    try:
        data = json.loads(bundle.read_text())
    except Exception as e:  # noqa: BLE001
        return [f"invalid JSON: {e}"]

    for e in sorted(VALIDATOR.iter_errors(data), key=lambda x: x.path):
        errs.append("schema: " + e.message)

    # ModernTSF binding (v1, lenient): the run must carry at least one handle
    # tying it to a ModernTSF run/config. The two submission shapes express it
    # differently, so accept any of them:
    #   - manifest bundle: datasets[].source_config (config path)
    #   - flat record:     env.git_sha (ModernTSF commit) or a config block
    bound = (
        any(ds.get("source_config") for ds in data.get("datasets", []))
        or bool((data.get("env") or {}).get("git_sha"))
        or bool(data.get("config"))
    )
    if not bound:
        errs.append(
            "ModernTSF binding missing: need datasets[].source_config, env.git_sha, or config"
        )
    return errs


def main() -> int:
    subs = sorted((ROOT / "submissions").rglob("submission.json"))
    if not subs:
        print("no submissions found.")
        return 0

    bad = 0
    for s in subs:
        errs = check(s)
        rel = s.relative_to(ROOT)
        if errs:
            bad += 1
            print(f"✗ {rel}")
            for e in errs:
                print(f"    - {e}")
        # (silent on success to keep CI logs readable)

    ok = len(subs) - bad
    print(f"\n{ok}/{len(subs)} submissions valid.")
    if bad:
        print(f"❌ {bad} invalid submission(s).")
        return 1
    print("✅ all submissions valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
