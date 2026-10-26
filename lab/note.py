"""Create an experiment journal entry: python -m lab.note <slug> --title 'Hypothesis'."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("--title", required=True)
    parser.add_argument("--run", default="", help="sweep directory relative to results/")
    parser.add_argument("--profile", default="", help="profile directory relative to results/")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        parser.error("slug must contain lowercase letters, digits and hyphens")
    path = ROOT / "docs/experiments" / f"{args.slug}.md"
    metadata = {"title": args.title, "description": "Evidence, hypothesis, measurements and decision.",
                "date": date.today().isoformat(), "status": "planned", "run": args.run, "profile": args.profile}
    body = (ROOT / "docs/experiments/_template.md").read_text()
    with path.open("x") as stream:
        stream.write("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + body)
    print(path)


if __name__ == "__main__":
    main()
