from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from scripts.results_writer import make_results_csv_path, write_csv


class HeaParseError(ValueError):
	pass

@dataclass(frozen=True)
class HeaRecordSummary:
	record_name: str
	n_signals: int
	fs_hz: Optional[float]
	n_samples: Optional[int]
	extra_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class HeaSignalSummary:
	index: int
	file_name: Optional[str]
	fmt: Optional[str]
	gain: Optional[float]
	bit_resolution: Optional[int]
	units: Optional[str]
	description: Optional[str]
	raw_line: str


class HeaAnalyzer:
	def __init__(self, hea_path: str, *, max_comment_lines: int = 200) -> None:
		self.hea_path = hea_path
		self.max_comment_lines = max_comment_lines

		self._lines: List[str] = []
		self._record: Optional[HeaRecordSummary] = None
		self._signals: List[HeaSignalSummary] = []
		self._comments: List[str] = []

	def load(self) -> None:
		path = Path(self.hea_path)
		if not path.exists():
			raise FileNotFoundError(str(path))
		if not path.is_file():
			raise HeaParseError(f"Not a file: {path}")

		data = path.read_bytes()
		try:
			text = data.decode("utf-8")
		except UnicodeDecodeError:
			text = data.decode("latin-1", errors="replace")

		self._lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
		self._parse()

	def record_summary(self) -> HeaRecordSummary:
		if self._record is None:
			self.load()
		if self._record is None:
			raise HeaParseError("Missing record header")
		return self._record

	def signal_summaries(self) -> List[HeaSignalSummary]:
		if self._record is None:
			self.load()
		return list(self._signals)

	def build_report(self) -> str:
		if self._record is None:
			self.load()

		record = self.record_summary()
		signals = self.signal_summaries()

		lines: List[str] = []
		lines.append("=" * 80)
		lines.append(f"HEA Analysis Report: {self.hea_path}")
		lines.append("=" * 80)
		lines.append("")

		lines.append("Record")
		lines.append("-" * 80)
		lines.append(f"record_name: {record.record_name}")
		lines.append(f"n_signals : {record.n_signals}")
		lines.append(f"fs_hz     : {record.fs_hz if record.fs_hz is not None else '-'}")
		lines.append(f"n_samples : {record.n_samples if record.n_samples is not None else '-'}")
		if record.extra_tokens:
			lines.append(f"extra     : {' '.join(record.extra_tokens)}")
		lines.append("")

		lines.append("Signals")
		lines.append("-" * 80)
		if not signals:
			lines.append("(No signal lines parsed.)")
		for s in signals:
			lines.append(
				f"[{s.index}] file={s.file_name or '-'} fmt={s.fmt or '-'} units={s.units or '-'}"
			)
			gain_str = f"{s.gain}" if s.gain is not None else "-"
			bit_str = f"{s.bit_resolution}" if s.bit_resolution is not None else "-"
			lines.append(f"  gain={gain_str} bit_resolution={bit_str}")
			if s.description:
				lines.append(f"  desc={s.description}")
		lines.append("")

		dat_path = Path(self.hea_path).with_suffix(".dat")
		lines.append("Related Files")
		lines.append("-" * 80)
		lines.append(f".dat exists: {dat_path.exists()}  ({dat_path})")
		lines.append("")

		if self._comments:
			lines.append("Header Comments")
			lines.append("-" * 80)
			preview = self._comments[: self.max_comment_lines]
			lines.extend(preview)
			if len(self._comments) > len(preview):
				lines.append(f"... (+{len(self._comments) - len(preview)} more)")
			lines.append("")

		return "\n".join(lines)

	def print_report(self) -> None:
		print(self.build_report())

	def _parse(self) -> None:
		self._record = None
		self._signals = []
		self._comments = []

		content = [ln.strip() for ln in self._lines if ln.strip()]
		if not content:
			raise HeaParseError("Empty .hea file")

		record_line: Optional[str] = None
		record_index: Optional[int] = None
		for i, ln in enumerate(content):
			if ln.startswith("#"):
				self._comments.append(ln)
				continue
			record_line = ln
			record_index = i
			break

		if record_line is None or record_index is None:
			raise HeaParseError("No record header line found")

		self._record = self._parse_record_line(record_line)

		signal_lines: List[str] = []
		for ln in content[record_index + 1 :]:
			if ln.startswith("#"):
				self._comments.append(ln)
				continue
			signal_lines.append(ln)
			if len(signal_lines) >= self._record.n_signals:
				break

		for idx, ln in enumerate(signal_lines, start=1):
			self._signals.append(self._parse_signal_line(idx, ln))

	def _parse_record_line(self, line: str) -> HeaRecordSummary:
		parts = line.split()
		if len(parts) < 2:
			raise HeaParseError(f"Invalid record line: {line!r}")

		record_name = parts[0]
		try:
			n_signals = int(parts[1])
		except ValueError as exc:
			raise HeaParseError(f"Invalid n_signals in record line: {line!r}") from exc

		fs_hz: Optional[float] = self._parse_float(parts[2]) if len(parts) >= 3 else None
		n_samples: Optional[int] = self._parse_int(parts[3]) if len(parts) >= 4 else None
		extra = tuple(parts[4:]) if len(parts) > 4 else tuple()

		return HeaRecordSummary(
			record_name=record_name,
			n_signals=n_signals,
			fs_hz=fs_hz,
			n_samples=n_samples,
			extra_tokens=extra,
		)

	def _parse_signal_line(self, index: int, line: str) -> HeaSignalSummary:
		parts = line.split()
		file_name = parts[0] if len(parts) >= 1 else None
		fmt = parts[1] if len(parts) >= 2 else None

		gain: Optional[float] = None
		bit_resolution: Optional[int] = None
		units: Optional[str] = None
		description: Optional[str] = None

		if len(parts) >= 3:
			gain, bit_resolution = self._parse_gain_bitres(parts[2])
		if len(parts) >= 4:
			units = parts[3]
		if len(parts) >= 9:
			description = " ".join(parts[8:]).strip() or None

		return HeaSignalSummary(
			index=index,
			file_name=file_name,
			fmt=fmt,
			gain=gain,
			bit_resolution=bit_resolution,
			units=units,
			description=description,
			raw_line=line,
		)

	def _parse_gain_bitres(self, token: str) -> tuple[Optional[float], Optional[int]]:
		core = token.split("(", 1)[0]
		if "/" in core:
			left, right = core.split("/", 1)
			return self._parse_float(left), self._parse_int(right)
		return self._parse_float(core), None

	def _parse_float(self, token: str) -> Optional[float]:
		t = token.split("/", 1)[0]
		try:
			return float(t)
		except Exception:
			return None

	def _parse_int(self, token: str) -> Optional[int]:
		t = token.split("/", 1)[0]
		try:
			return int(float(t))
		except Exception:
			return None


class HeaFileReader:
	def __init__(self, path_way: str):
		self.path_way = path_way

	def choosed_file_reader(self) -> None:
		analyzer = HeaAnalyzer(self.path_way)
		report = analyzer.build_report()
		print(report)

		csv_path = make_results_csv_path(self.path_way, kind="hea")
		record = analyzer.record_summary()
		rows = [
			{
				"row_type": "record",
				"record_name": record.record_name,
				"n_signals": record.n_signals,
				"fs_hz": record.fs_hz,
				"n_samples": record.n_samples,
			}
		]

		for s in analyzer.signal_summaries():
			rows.append(
				{
					"row_type": "signal",
					"record_name": record.record_name,
					"fmt": s.fmt,
					"gain": s.gain,
					"bit_resolution": s.bit_resolution,
					"units": s.units,
					"description": s.description,
					"raw_line": s.raw_line,
				}
			)

		write_csv(
			csv_path,
			fieldnames=(
				"row_type",
				"record_name",
				"n_signals",
				"fs_hz",
				"n_samples",
				"fmt",
				"gain",
				"bit_resolution",
				"units",
				"description",
				"raw_line",
			),
			rows=rows,
		)
		print(f"CSV saved: {csv_path}")
