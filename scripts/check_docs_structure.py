#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_MD = {
    "README.md",
    "AGENTS.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
}
REQUIRED_FILES = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "archive" / "README.md",
    ROOT / "docs" / "getting-started" / "OPERATOR_QUICKSTART.md",
    ROOT / "docs" / "datasets" / "DATASET_ONBOARDING_GUIDE.md",
    ROOT / "docs" / "architecture" / "CURRENT_SYSTEM.md",
    ROOT / "docs" / "operations" / "STATUS.md",
    ROOT / "docs" / "reports" / "faz11" / "FINAL_REPORT.md",
    ROOT / "docs" / "agents" / "START_HERE.md",
    ROOT / "docs" / "agents" / "AGENT_INSTRUCTIONS.md",
    ROOT / "docs" / "agents" / "TASKS.md",
    ROOT / "docs" / "agents" / "CONTEXT.md",
    ROOT / "docs" / "agents" / "WEB_CHAT_HANDOFF.md",
    ROOT / "docs" / "agents" / "prompts" / "README.md",
]
LIVE_DOC_DIRS = [
    ROOT / "docs",
]
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def iter_markdown_files() -> Iterable[Path]:
    for path in ROOT.rglob("*.md"):
        if ".venv" in path.parts or ".testdeps" in path.parts or ".runtime" in path.parts:
            continue
        if any(part in {"site-packages", "node_modules"} for part in path.parts):
            continue
        if "docs" in path.parts and "archive" in path.parts:
            continue
        yield path


def check_root_markdown() -> list[str]:
    errors: list[str] = []
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix == ".md" and path.name not in ALLOWED_ROOT_MD:
            errors.append(f"Forbidden root markdown file: {path.relative_to(ROOT)}")
    return errors


def check_required_files() -> list[str]:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing required file: {path.relative_to(ROOT)}")
    return errors


def check_links_in_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return errors

    for _, target in LINK_PATTERN.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean_target = target.split("#", 1)[0].split("?", 1)[0]
        if not clean_target:
            continue
        target_path = (path.parent / clean_target).resolve()
        if not target_path.exists():
            errors.append(f"Broken link in {path.relative_to(ROOT)} -> {target}")
    return errors


def check_live_docs_links() -> list[str]:
    errors: list[str] = []
    for doc in iter_markdown_files():
        errors.extend(check_links_in_file(doc))
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_root_markdown())
    errors.extend(check_required_files())
    errors.extend(check_live_docs_links())

    if errors:
        for error in errors:
            print(error)
        return 1

    print("docs structure ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
