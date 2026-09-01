from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_analysis_plot(df, sparse_matrix, output_path="output/analysis.png"):
    """Generate the analysis figure and save it to disk."""
    plot_correlation_heatmap(df, sparse_matrix, output_path)


def plot_correlation_heatmap(df, sparse_matrix, output_path):
    '''Generates a combined figure with a correlation heatmap and a sparse matrix structure plot.'''
    # Ensure the target directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up a 1-row, 2-column plot figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Scientific Data Explorer - Visual Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Feature Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            ax=axes[0],
            cbar=True
        )
        axes[0].set_title('Numeric Feature Correlations')
    else:
        axes[0].text(
            0.5,
            0.5,
            'No numeric data available',
            ha='center',
            va='center',
            fontsize=12,
        )
        axes[0].set_title('Feature Correlations')

    # Plot 2: SciPy Sparse Matrix Non-Zero Pattern (plt.spy)
    axes[1].spy(sparse_matrix, markersize=2, color='crimson')
    axes[1].set_title('Sparse Matrix Structure (Non-Zero Pattern)')
    axes[1].set_xlabel('Columns')
    axes[1].set_ylabel('Rows')

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)