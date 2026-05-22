from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / 'results' / 'results_expletives.csv'
OUTPUT_DIR = BASE_DIR / 'results'

EXCLUDED_WORDS = {'wang', 'dick', 'willy', 'cox'}
TOP_N = 5


def build_chart(df: pd.DataFrame, output_path: Path) -> None:
    linus_top = df.sort_values('Linus_Count', ascending=False).head(TOP_N)[['Expletive', 'Linus_Count']]
    greg_top = df.sort_values('Greg_Count', ascending=False).head(TOP_N)[['Expletive', 'Greg_Count']]

    terms = list(dict.fromkeys(list(linus_top['Expletive']) + list(greg_top['Expletive'])))

    if not terms:
        raise ValueError('No data available to plot.')

    plot_df = pd.DataFrame({'Expletive': terms})
    plot_df = plot_df.merge(linus_top, on='Expletive', how='left').merge(greg_top, on='Expletive', how='left')
    plot_df = plot_df.fillna(0)

    positions = np.arange(len(plot_df))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))

    linus_bars = ax.barh(
        positions - bar_height / 2,
        plot_df['Linus_Count'],
        height=bar_height,
        color='#1f77b4',
        label='Linus Torvalds',
    )
    greg_bars = ax.barh(
        positions + bar_height / 2,
        plot_df['Greg_Count'],
        height=bar_height,
        color='#ff7f0e',
        label='Greg Kroah-Hartman',
    )

    for bars in (linus_bars, greg_bars):
        for bar in bars:
            width = bar.get_width()
            if width <= 0:
                continue
            ax.annotate(
                f'{int(width)}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords='offset points',
                va='center',
                ha='left',
                fontsize=9,
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_df['Expletive'])
    ax.invert_yaxis()
    ax.set_xlabel('Occurrences')
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def filter_expletives(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[~filtered['Expletive'].isin(EXCLUDED_WORDS)]
    filtered.loc[filtered['Expletive'] == 'dick', 'Linus_Count'] = 0
    return filtered


df = pd.read_csv(INPUT_PATH)
df = df[df['Total_Count'] > 0].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

build_chart(
    df,
    OUTPUT_DIR / 'expletives_bar_normal.png',
)

build_chart(
    filter_expletives(df),
    OUTPUT_DIR / 'expletives_bar_filtered.png',
)