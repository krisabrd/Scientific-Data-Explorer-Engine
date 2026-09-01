from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from data_loader import clean_data, extract_numeric_matrix, get_dataset_summary, load_dataset
from matrix_ops import benchmark_multiplication, calculate_sparsity, compare_memory, to_sparse_csr
from viz import generate_analysis_plot

console = Console()


def resolve_input_path(file_path):
    path = Path(file_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / path
    return path.resolve()


def run_pipeline(file_path):
    '''Executes the complete scientific data exploration pipeline.'''
    path = resolve_input_path(file_path)

    # 1. Load Data
    console.print(f'\n[bold blue]Loading dataset from:[/bold blue] {path}')
    df_raw = load_dataset(path)

    # 2. Dataset Summary Table
    summary = get_dataset_summary(df_raw)
    table = Table(title='Dataset Summary', show_header=True, header_style='bold magenta')
    table.add_column('Metric', style='cyan')
    table.add_column('Value', style='green')

    table.add_row('Total Rows', str(summary['rows']))
    table.add_row('Total Columns', str(summary['columns']))
    table.add_row('Missing Values (Nulls)', str(summary['total_nulls']))
    table.add_row('Memory Size (MB)', f"{summary['memory_mb']} MB")
    console.print(table)

    # 3. Clean & Extract Matrix
    with console.status('[bold green]Cleaning data and extracting numeric matrix...'):
        df_cleaned = clean_data(df_raw, fill_value=0.0)
        dense_array = extract_numeric_matrix(df_cleaned)

    # 4. Matrix Operations & Benchmarks
    with console.status('[bold green]Running SciPy sparse conversion and benchmarks...'):
        sparse_matrix = to_sparse_csr(dense_array, threshold=0.0)
        sparsity = calculate_sparsity(dense_array)
        mem_stats = compare_memory(dense_array, sparse_matrix)
        benchmarks = benchmark_multiplication(dense_array, sparse_matrix)

    # 5. Display Benchmark Results Panel
    results_text = (
        f"[bold]Matrix Sparsity:[/bold] {sparsity}%\n\n"
        f"[bold cyan]Dense Memory:[/bold cyan] {mem_stats['dense_kb']} KB\n"
        f"[bold cyan]Sparse Memory:[/bold cyan] {mem_stats['sparse_kb']} KB\n"
        f"[bold yellow]RAM Savings:[/bold yellow] [green]{mem_stats['savings_percent']}%[/green]\n\n"
        f"[bold cyan]Dense Multiply Time:[/bold cyan] {benchmarks['dense_time_ms']} ms\n"
        f"[bold cyan]Sparse Multiply Time:[/bold cyan] {benchmarks['sparse_time_ms']} ms\n"
        f"[bold yellow]Performance Speedup:[/bold yellow] [green]{benchmarks['speedup_factor']}x faster[/green]"
    )
    console.print(Panel(results_text, title="Dense vs. Sparse Performance Engine", expand=False))

    # 6. Generate Visual Exports
    output_path = Path(__file__).resolve().parents[2] / 'output' / 'analysis.png'
    with console.status("[bold green]Exporting figures to output/analysis.png..."):
        generate_analysis_plot(df_cleaned, sparse_matrix, str(output_path))

    console.print("\n[bold green]✔ Pipeline complete![/bold green] Analysis figure exported to [underline]output/analysis.png[/underline]\n")


def main():
    """Main CLI entry point."""
    console.print("[bold yellow]============================================[/bold yellow]")
    console.print("[bold yellow]     SCIENTIFIC DATA EXPLORER ENGINE       [/bold yellow]")
    console.print("[bold yellow]============================================[/bold yellow]")

    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / 'data' / 'sample.csv'
    user_path = Prompt.ask("\nEnter path to CSV dataset", default=str(default_path))

    try:
        run_pipeline(user_path)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()