# dataset_analysis

A CLI-based dataset exploration (profiling) tool. It analyzes a user-selected `.mat` or `.hea` file, prints a summary report to the terminal, and saves results as CSV files under the `results/` folder.

This project is focused on quickly exploring what’s inside a `.mat` file, identifying likely feature matrices and diagnostic/label information, and getting a fast view of class distributions.

## Features

- **`.mat` analysis**
  - Variable listing (excluding MATLAB internal `__*` keys)
  - For each variable: `type`, `dtype`, `shape`, `ndim`, `size`, short `preview`
  - Heuristic candidates:
    - Feature matrix candidates (numeric, 2D, high sample axis)
    - Label/diagnostic candidates (1D, Nx1/1xN, small shapes, categorical/string-like)
  - Label distributions (unique value counts) and a simple label→class-name mapping suggestion

- **`.hea` analysis**
  - Parses the HEA header to extract a record/signal summary (e.g., `record_name`, `n_signals`, `fs_hz`, `n_samples`)
  - Produces a terminal report

- **CSV output**
  - Automatically saves analysis results to the `results/` folder
  - Fixed file naming: `filename_mat.csv` or `filename_hea.csv` (re-analyzing the same input overwrites the same CSV)

## Project Structure

- Entry point: `__main__.py`
- App flow: `scripts/start_app.py` → `scripts/process_management.py` → `scripts/path_management.py`
- Operations:
  - `scripts/operations/mat_file_reader.py`
  - `scripts/operations/hea_file_reader.py`
- CSV writing / results folder management: `scripts/results_writer.py`

## Setup

### 1) Virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Dependencies

- **SciPy** is required to read `.mat` files.

```bash
pip install scipy numpy
```

Note: No extra package is required for `.hea` parsing (the header is parsed as text).

## Running

From the project root:

```bash
python __main__.py
```

Flow:

1. The app shows a menu.
2. Choose an operation (`mat_file_reader.py` or `hea_file_reader.py`).
3. Browse directories and select a file with the expected extension (`.mat` / `.hea`).
4. A report is printed to the terminal and a CSV is written under `results/`.

## Outputs

### results/

Analysis outputs are written automatically to:

- `results/`

File naming:

- For `.mat`: `<filename>_mat.csv`
- For `.hea`: `<filename>_hea.csv`

## CSV Schemas

### MAT (`*_mat.csv`)

Each row represents a variable summary:

- `row_type` (constant: `variable`)
- `name`
- `py_type`
- `dtype`
- `shape`
- `ndim`
- `size`
- `preview`

### HEA (`*_hea.csv`)

Record row(s) + signal row(s) are stored in the same file:

- `row_type` (`record` or `signal`)
- `record_name`
- `n_signals`
- `fs_hz`
- `n_samples`
- `fmt`
- `gain`
- `bit_resolution`
- `units`
- `description`
- `raw_line`

Note: Path/filename/extra/index fields are intentionally not stored in the CSV.

## Known Limitations

- If a `.mat` variable is very large, the `preview` may be truncated.
- The `.hea` side only parses the header; it does not decode `.dat` signal data.
- Feature/label candidate detection is **heuristic** and may need dataset-specific tuning.

## Development Notes

- To add a new file type, add a new reader under `scripts/operations/` and wire it in the operation dispatch in `scripts/process_management.py`.
- CSV naming and writing are managed in `scripts/results_writer.py`.

## License

No license information is included in this repository. Add a LICENSE file if needed.
