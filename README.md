# Scientific Data Explorer Engine

A modular Python CLI tool designed to ingest, clean, and process scientific datasets, run performance comparisons between dense NumPy arrays and SciPy CSR sparse matrices, and visualize output structures. 

*Planned Future Expansion: Transforming this core engine into a full-stack web application for interactive dataset analysis.*

---

## 🛠 Project Architecture & File Breakdown

* **`data/sample.csv`**: Synthetic testing dataset populated with sparse floating-point numeric data used to validate cleaning and matrix conversion routines.
* **`backend/src/main.py`**: The CLI entry point. Uses `Rich` to handle interactive user prompts, display tabular dataset summaries, present benchmark results in styled panels, and coordinate execution across modules.
* **`backend/src/data_loader.py`**: Manages cross-platform path resolution, ingests raw CSV files with Pandas, handles null-filling operations, extracts numeric matrices, and generates summary metrics.
* **`backend/src/matrix_ops.py`**: Converts dense NumPy arrays to SciPy CSR sparse matrices, calculates dataset sparsity percentages, performs memory footprint comparisons, and benchmarks matrix-vector multiplication performance.
* **`backend/src/viz.py`**: Uses Matplotlib and Seaborn to generate side-by-side heatmaps and spatial coordinate plots of the matrix structure, saving the output to `output/analysis.png`.
* **`tests/test_data_loader.py`**: Automated unit tests using `pytest` to verify dataset loading, missing value handling, and path resolution integrity.
* **`tests/test_matrix_ops.py`**: Automated unit tests verifying sparse CSR conversion thresholds, sparsity calculation precision, and matrix multiplication benchmark logic.

---

## 💡 Key Learnings & Technical Takeaways

* **Path Resolution (`pathlib`):** Mastered dynamic path resolution using `Path(__file__).resolve().parents[2]` to construct absolute file paths, ensuring cross-platform stability regardless of execution context.
* **Sparse Matrix Math (`SciPy`):** Explored compressed sparse row (`CSR`) representation with `scipy.sparse`, measuring actual RAM reduction and speed gains during matrix operations.
* **Data Visualization (`Matplotlib` / `Seaborn`):** Exported clean, structured dual-panel figures with `matplotlib` and `seaborn`, saving the output as a PNG for downstream analysis and reporting.
* **Expanded Pandas Expertise:** Reinforced data ingestion routines, handling null values safely, and cleanly extracting pure numerical matrices from DataFrames.

---
## 🚀 Getting Started

### 1. Environment Setup
Install the required dependencies:

```bash
pip install pandas numpy scipy matplotlib seaborn rich pytest
```

### 2. Running the CLI
From the project root, run:
```bash
cd backend/src
python main.py
```
You can then enter the default dataset path or provide a custom CSV path. The app resolves relative paths safely and will use the sample dataset in `data/sample.csv` by default.

### 3. Running Test Suite
Run the test suite from the project root:
```bash
python -m pytest
```

> Note: The project is set up so the app and tests can resolve imports correctly when run from the repository root or from the backend directory context described above.
