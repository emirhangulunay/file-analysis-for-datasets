from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.results_writer import make_results_csv_path, write_csv

try:
    import scipy.io as _scipy_io
except ModuleNotFoundError:
    _scipy_io = None


@dataclass(frozen=True)
class VariableSummary:
    name: str
    py_type: str
    dtype: Optional[str]
    shape: Optional[Tuple[int, ...]]
    ndim: Optional[int]
    size: Optional[int]
    preview: str


class MatAnalyzer:
    def __init__(
        self,
        mat_path: str,
        *,
        preview_rows: int = 5,
        sample_threshold: int = 200,
        max_preview_chars: int = 2000,
    ) -> None:
        self.mat_path = mat_path
        self.preview_rows = preview_rows
        self.sample_threshold = sample_threshold
        self.max_preview_chars = max_preview_chars

        self._mat: Dict[str, Any] = {}
        self._keys: List[str] = []

    def load(self) -> Dict[str, Any]:
        if _scipy_io is None:
            raise RuntimeError(
                "SciPy is required to read .mat files. Install it with: pip install scipy"
            )
        try:
            self._mat = _scipy_io.loadmat(
                self.mat_path,
                squeeze_me=True,
                struct_as_record=False,
            )
        except TypeError:
            self._mat = _scipy_io.loadmat(self.mat_path)

        self._keys = sorted(k for k in self._mat.keys() if not k.startswith("__"))
        return self._mat

    def list_keys(self) -> List[str]:
        if not self._mat:
            self.load()
        return list(self._keys)

    def summarize_variables(self) -> List[VariableSummary]:
        if not self._mat:
            self.load()

        summaries: List[VariableSummary] = []
        for key in self._keys:
            value = self._mat.get(key)
            summaries.append(self._summarize_one(key, value))
        return summaries

    def detect_feature_candidates(self) -> List[str]:
        if not self._mat:
            self.load()

        candidates: List[Tuple[str, int]] = []
        for key in self._keys:
            arr = self._as_ndarray(self._mat.get(key))
            if arr is None:
                continue
            if arr.ndim != 2:
                continue
            if not self._is_numeric_ndarray(arr):
                continue

            n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
            if min(n_rows, n_cols) < 2:
                continue

            sample_like = max(n_rows, n_cols)
            if sample_like >= self.sample_threshold:
                candidates.append((key, sample_like))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates]

    def detect_label_candidates(self) -> List[str]:
        if not self._mat:
            self.load()

        candidates: List[Tuple[str, int]] = []
        for key in self._keys:
            value = self._mat.get(key)
            arr = self._as_ndarray(value)
            if arr is None:
                if self._looks_like_class_names(value):
                    candidates.append((key, 0))
                continue

            shape = arr.shape
            if arr.ndim == 1:
                candidates.append((key, int(shape[0])))
                continue

            if arr.ndim == 2:
                n_rows, n_cols = int(shape[0]), int(shape[1])
                if min(n_rows, n_cols) == 1:
                    candidates.append((key, max(n_rows, n_cols)))
                    continue
                if arr.size <= 1000 and max(n_rows, n_cols) <= 50:
                    candidates.append((key, arr.size))
                    continue

            if self._looks_like_string_array(arr):
                candidates.append((key, arr.size))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates]

    def label_distributions(self) -> Dict[str, Dict[Any, int]]:
        if not self._mat:
            self.load()

        dists: Dict[str, Dict[Any, int]] = {}
        for key in self.detect_label_candidates():
            arr = self._as_ndarray(self._mat.get(key))
            if arr is None:
                continue

            vec = self._to_1d_vector(arr)
            if vec is None:
                continue

            if np.issubdtype(vec.dtype, np.floating):
                finite = vec[np.isfinite(vec)]
                if finite.size > 0:
                    rounded = np.rint(finite)
                    if np.allclose(finite, rounded):
                        vec = np.rint(vec).astype(np.int64, copy=False)

            if np.issubdtype(vec.dtype, np.floating):
                vec = vec[np.isfinite(vec)]
            uniques, counts = np.unique(vec, return_counts=True)
            dists[key] = {self._to_py_scalar(u): int(c) for u, c in zip(uniques, counts)}
        return dists

    def find_class_name_candidates(self) -> Dict[str, List[str]]:
        if not self._mat:
            self.load()

        out: Dict[str, List[str]] = {}
        for key in self._keys:
            value = self._mat.get(key)
            names = self._extract_strings(value)
            if names and 1 <= len(names) <= 500:
                out[key] = names
        return out

    def suggest_label_name_mapping(self) -> Dict[str, Dict[Any, str]]:
        class_name_vars = self.find_class_name_candidates()
        if not class_name_vars:
            return {}

        mappings: Dict[str, Dict[Any, str]] = {}
        dists = self.label_distributions()
        for label_var, dist in dists.items():
            labels_sorted = sorted(dist.keys())
            if not labels_sorted:
                continue

            # Only attempt for integer-like labels
            if not all(isinstance(x, (int, np.integer)) for x in labels_sorted):
                continue

            unique_labels = [int(x) for x in labels_sorted]
            k = len(unique_labels)

            best: Optional[Tuple[str, Dict[Any, str]]] = None
            for name_var, names in class_name_vars.items():
                if len(names) != k:
                    continue

                if unique_labels == list(range(1, k + 1)):
                    mapping = {lab: names[i] for i, lab in enumerate(unique_labels)}
                    best = (name_var, mapping)
                    break
                if unique_labels == list(range(0, k)):
                    mapping = {lab: names[lab] for lab in unique_labels}
                    best = (name_var, mapping)
                    break

            if best is not None:
                _, mapping = best
                mappings[label_var] = mapping
        return mappings

    def build_report(self) -> str:
        if not self._mat:
            self.load()

        summaries = self.summarize_variables()
        feature_candidates = self.detect_feature_candidates()
        label_candidates = self.detect_label_candidates()
        label_dists = self.label_distributions()
        label_name_map = self.suggest_label_name_mapping()

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(f"MAT Analysis Report: {self.mat_path}")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Variables (filtered, excluding '__*'): {len(self._keys)}")
        for k in self._keys:
            lines.append(f"- {k}")
        lines.append("")

        lines.append("Variable Details")
        lines.append("-" * 80)
        for s in summaries:
            shape_str = "None" if s.shape is None else str(tuple(int(x) for x in s.shape))
            dtype_str = s.dtype or "-"
            lines.append(f"[{s.name}]")
            lines.append(f"  type : {s.py_type}")
            lines.append(f"  dtype: {dtype_str}")
            lines.append(f"  shape: {shape_str}")
            lines.append(f"  ndim : {s.ndim if s.ndim is not None else '-'}")
            lines.append(f"  size : {s.size if s.size is not None else '-'}")
            lines.append("  preview:")
            for pl in s.preview.splitlines():
                lines.append(f"    {pl}")
            lines.append("")

        lines.append("Auto-Detected Candidates")
        lines.append("-" * 80)
        lines.append("Feature matrix candidates (numeric, 2D, high sample axis):")
        if feature_candidates:
            for name in feature_candidates:
                arr = self._as_ndarray(self._mat.get(name))
                lines.append(f"- {name}  shape={tuple(int(x) for x in arr.shape)}")
        else:
            lines.append("- (none found by heuristic)")
        lines.append("")

        lines.append("Label/diagnostic candidates (1D, Nx1/1xN, small shapes, or categorical):")
        if label_candidates:
            for name in label_candidates:
                arr = self._as_ndarray(self._mat.get(name))
                shape_str = "None" if arr is None else str(tuple(int(x) for x in arr.shape))
                lines.append(f"- {name}  shape={shape_str}")
        else:
            lines.append("- (none found by heuristic)")
        lines.append("")

        lines.append("Label Distributions")
        lines.append("-" * 80)
        if label_dists:
            for name, dist in label_dists.items():
                lines.append(f"[{name}] classes={len(dist)}")
                for lab in sorted(dist.keys(), key=lambda x: (str(type(x)), str(x))):
                    count = dist[lab]
                    if name in label_name_map and lab in label_name_map[name]:
                        lines.append(f"  {lab}: {count}  -> {label_name_map[name][lab]}")
                    else:
                        lines.append(f"  {lab}: {count}")
                lines.append("")
        else:
            lines.append("(No label distributions computed.)")
            lines.append("")

        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.build_report())

    def _summarize_one(self, name: str, value: Any) -> VariableSummary:
        py_type = type(value).__name__
        arr = self._as_ndarray(value)

        dtype: Optional[str] = None
        shape: Optional[Tuple[int, ...]] = None
        ndim: Optional[int] = None
        size: Optional[int] = None

        if arr is not None:
            dtype = str(arr.dtype)
            shape = tuple(int(x) for x in arr.shape)
            ndim = int(arr.ndim)
            size = int(arr.size)
            preview = self._preview_ndarray(arr)
        else:
            preview = self._truncate(self._safe_repr(value))

        return VariableSummary(
            name=name,
            py_type=py_type,
            dtype=dtype,
            shape=shape,
            ndim=ndim,
            size=size,
            preview=preview,
        )

    def _as_ndarray(self, value: Any) -> Optional[np.ndarray]:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, np.generic):
            return np.asarray(value)
        return None

    def _is_numeric_ndarray(self, arr: np.ndarray) -> bool:
        return np.issubdtype(arr.dtype, np.number) and arr.dtype != np.dtype("object")

    def _looks_like_string_array(self, arr: np.ndarray) -> bool:
        if arr.dtype.kind in {"U", "S"}:
            return True
        if arr.dtype == np.dtype("object"):
            flat = arr.ravel()
            sample = flat[: min(50, flat.size)]
            return any(isinstance(x, str) for x in sample)
        return False

    def _to_1d_vector(self, arr: np.ndarray) -> Optional[np.ndarray]:
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and 1 in arr.shape:
            return arr.reshape(-1)
        return None

    def _preview_ndarray(self, arr: np.ndarray) -> str:
        if self._looks_like_string_array(arr):
            names = self._extract_strings(arr)
            if names:
                head = names[: self.preview_rows]
                more = "" if len(names) <= self.preview_rows else f" ... (+{len(names) - self.preview_rows} more)"
                return self._truncate("\n".join(["strings:"] + [f"- {x}" for x in head]) + more)

        if arr.ndim == 0:
            return self._truncate(self._safe_repr(self._to_py_scalar(arr)))

        with np.printoptions(
            edgeitems=3,
            linewidth=120,
            suppress=True,
            threshold=50,
        ):
            if arr.ndim == 1:
                head = arr[: self.preview_rows]
                return self._truncate(np.array2string(head))
            if arr.ndim == 2:
                head = arr[: self.preview_rows, :]
                return self._truncate(np.array2string(head))

            slicer = (slice(0, self.preview_rows),) + (slice(None),) * (arr.ndim - 1)
            head = arr[slicer]
            return self._truncate(f"ndarray ndim={arr.ndim} head slice:\n" + np.array2string(head))

    def _extract_strings(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, (list, tuple)):
            out: List[str] = []
            for v in value:
                out.extend(self._extract_strings(v))
            return [x for x in out if x != ""]

        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S"}:
                flat = value.ravel()
                out = []
                for x in flat:
                    s = x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, np.bytes_)) else str(x)
                    s = s.strip()
                    if s:
                        out.append(s)
                return out
            if value.dtype == np.dtype("object"):
                flat = value.ravel()
                out = []
                for x in flat:
                    if isinstance(x, (bytes, np.bytes_)):
                        s = x.decode("utf-8", errors="ignore").strip()
                        if s:
                            out.append(s)
                    elif isinstance(x, str):
                        s = x.strip()
                        if s:
                            out.append(s)
                    else:
                        out.extend(self._extract_strings(x))
                return out

        return []

    def _looks_like_class_names(self, value: Any) -> bool:
        names = self._extract_strings(value)
        return len(names) >= 2

    def _safe_repr(self, value: Any) -> str:
        try:
            return repr(value)
        except Exception:
            return f"<{type(value).__name__} repr failed>"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_preview_chars:
            return text
        return text[: self.max_preview_chars] + " ... (truncated)"

    def _to_py_scalar(self, x: Any) -> Any:
        if isinstance(x, np.generic):
            return x.item()
        return x


class MatFileReader:
    def __init__(self, path_way: str):
        self.path_way = path_way

    def choosed_file_reader(self) -> None:
        analyzer = MatAnalyzer(self.path_way)
        report = analyzer.build_report()
        print(report)

        csv_path = make_results_csv_path(self.path_way, kind="mat")
        rows = []
        for s in analyzer.summarize_variables():
            rows.append(
                {
                    "row_type": "variable",
                    "name": s.name,
                    "py_type": s.py_type,
                    "dtype": s.dtype,
                    "shape": "" if s.shape is None else str(tuple(s.shape)),
                    "ndim": s.ndim,
                    "size": s.size,
                    "preview": s.preview,
                }
            )

        write_csv(
            csv_path,
            fieldnames=(
                "row_type",
                "name",
                "py_type",
                "dtype",
                "shape",
                "ndim",
                "size",
                "preview",
            ),
            rows=rows,
        )
        print(f"CSV saved: {csv_path}")
