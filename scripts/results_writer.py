from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ResultsPaths:
    project_root: Path
    results_dir: Path


def get_paths() -> ResultsPaths:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return ResultsPaths(project_root=project_root, results_dir=results_dir)


def make_results_csv_path(input_path: str, *, kind: str) -> Path:
    src = Path(input_path)
    safe_stem = src.stem or "result"
    paths = get_paths()
    return paths.results_dir / f"{safe_stem}_{kind}.csv"


def write_csv(path: Path, *, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
