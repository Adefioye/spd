from ast import literal_eval
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from koko_notebooks.shrinkage_analysis.shrinkage_analysis_common import PLOTS_DIR, ensure_outputs_dir


LAYER_COLORS = {
    "linear1": "#1f77b4",
    "hidden_layers.0": "#ff7f0e",
    "hidden_layers.1": "#2ca02c",
    "hidden_layers.2": "#d62728",
    "hidden_layers.3": "#9467bd",
    "linear2": "#8c564b",
}

ARCH_COLORS = {
    "tied": "#0b6e4f",
    "untied": "#b23a48",
}


def setup_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "figure.figsize": (7, 4.2),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "grid.color": "#a9a9a9",
        }
    )


def plot_output_dir(subdir: str) -> Path:
    ensure_outputs_dir()
    out_dir = PLOTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig: plt.Figure, subdir: str, stem: str) -> dict[str, str]:
    out_dir = plot_output_dir(subdir)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _series_color(name: str) -> str:
    if name in LAYER_COLORS:
        return LAYER_COLORS[name]
    if name in ARCH_COLORS:
        return ARCH_COLORS[name]
    return "#333333"


def line_plot_by_layer(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    subdir: str,
    stem: str,
    *,
    hline_at_one: bool = False,
    y_lim: tuple[float, float] | None = None,
) -> dict[str, str]:
    setup_publication_style()
    fig, ax = plt.subplots(constrained_layout=True)
    for layer_name, layer_df in df.groupby("layer_name", sort=False):
        ordered_df = layer_df.sort_values(x_col)
        ax.plot(
            ordered_df[x_col],
            ordered_df[y_col],
            marker="o",
            linewidth=2,
            markersize=4.5,
            label=layer_name,
            color=_series_color(layer_name),
        )
    if hline_at_one:
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, subdir=subdir, stem=stem)


def line_plot_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    subdir: str,
    stem: str,
    *,
    hline_at_one: bool = False,
    y_lim: tuple[float, float] | None = None,
) -> dict[str, str]:
    setup_publication_style()
    fig, ax = plt.subplots(constrained_layout=True)
    for group_name, group_df in df.groupby(group_col, sort=False):
        ordered_df = group_df.sort_values(x_col)
        ax.plot(
            ordered_df[x_col],
            ordered_df[y_col],
            marker="o",
            linewidth=2,
            markersize=4.5,
            label=group_name,
            color=_series_color(str(group_name)),
        )
    if hline_at_one:
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, subdir=subdir, stem=stem)


def multi_metric_panel_by_layer(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    titles: list[str],
    subdir: str,
    stem: str,
    *,
    hline_at_one: bool = False,
) -> dict[str, str]:
    setup_publication_style()
    fig, axes = plt.subplots(1, len(y_cols), figsize=(6.5 * len(y_cols), 4.2), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, y_col, title in zip(axes, y_cols, titles, strict=True):
        for layer_name, layer_df in df.groupby("layer_name", sort=False):
            ordered_df = layer_df.sort_values(x_col)
            ax.plot(
                ordered_df[x_col],
                ordered_df[y_col],
                marker="o",
                linewidth=2,
                markersize=4.5,
                label=layer_name,
                color=_series_color(layer_name),
            )
        if hline_at_one:
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_xlabel("Checkpoint")
        ax.set_title(title)
    axes[0].set_ylabel("Ratio")
    axes[-1].legend(frameon=False, ncol=1, loc="best")
    return save_figure(fig, subdir=subdir, stem=stem)


def multi_metric_panel_by_group(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    y_cols: list[str],
    titles: list[str],
    subdir: str,
    stem: str,
    *,
    hline_at_one: bool = False,
) -> dict[str, str]:
    setup_publication_style()
    fig, axes = plt.subplots(1, len(y_cols), figsize=(6.5 * len(y_cols), 4.2), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, y_col, title in zip(axes, y_cols, titles, strict=True):
        for group_name, group_df in df.groupby(group_col, sort=False):
            ordered_df = group_df.sort_values(x_col)
            ax.plot(
                ordered_df[x_col],
                ordered_df[y_col],
                marker="o",
                linewidth=2,
                markersize=4.5,
                label=group_name,
                color=_series_color(str(group_name)),
            )
        if hline_at_one:
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_title(title)
    axes[0].set_ylabel("Value")
    axes[-1].legend(frameon=False, ncol=1, loc="best")
    return save_figure(fig, subdir=subdir, stem=stem)


def architecture_comparison_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    subdir: str,
    stem: str,
    *,
    hline_at_one: bool = False,
) -> dict[str, str]:
    setup_publication_style()
    fig, ax = plt.subplots(constrained_layout=True)
    for architecture, arch_df in df.groupby("architecture", sort=False):
        ordered_df = arch_df.sort_values(x_col)
        ax.plot(
            ordered_df[x_col],
            ordered_df[y_col],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=architecture,
            color=_series_color(architecture),
        )
    if hline_at_one:
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return save_figure(fig, subdir=subdir, stem=stem)


def heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    subdir: str,
    stem: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    annotate: bool = False,
    fmt: str = ".2f",
) -> dict[str, str]:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(1.6 * len(col_labels) + 2.5, 0.7 * len(row_labels) + 2.5), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    if annotate:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                rgba = im.cmap(im.norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.55 else "#111111"
                ax.text(
                    col_idx,
                    row_idx,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    return save_figure(fig, subdir=subdir, stem=stem)


def histogram_triptych(
    arrays: list[np.ndarray],
    titles: list[str],
    super_title: str,
    subdir: str,
    stem: str,
    *,
    bins: int = 25,
) -> dict[str, str]:
    setup_publication_style()
    fig, axes = plt.subplots(1, len(arrays), figsize=(5.2 * len(arrays), 4), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, values, title in zip(axes, arrays, titles, strict=True):
        ax.hist(values.ravel(), bins=bins, color="#1f77b4", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.suptitle(super_title)
    return save_figure(fig, subdir=subdir, stem=stem)


def parse_vector_column(series: pd.Series) -> np.ndarray:
    parsed = []
    for value in series:
        if isinstance(value, list):
            parsed.append(np.asarray(value, dtype=float))
        else:
            parsed.append(np.asarray(literal_eval(value), dtype=float))
    return np.stack(parsed, axis=0)


def singular_value_trajectory_plot(
    df: pd.DataFrame,
    step_col: str,
    target_col: str,
    raw_col: str,
    title: str,
    subdir: str,
    stem: str,
    *,
    max_modes: int = 3,
) -> dict[str, str]:
    setup_publication_style()
    ordered_df = df.sort_values(step_col)
    target_values = parse_vector_column(ordered_df[target_col])
    raw_values = parse_vector_column(ordered_df[raw_col])
    n_modes = min(max_modes, target_values.shape[1], raw_values.shape[1])

    fig, ax = plt.subplots(constrained_layout=True)
    x = ordered_df[step_col].to_numpy()
    for mode_idx in range(n_modes):
        color = plt.cm.tab10(mode_idx)
        ax.plot(
            x,
            target_values[:, mode_idx],
            linewidth=2.2,
            color=color,
            label=f"target $\\sigma_{mode_idx + 1}$",
        )
        ax.plot(
            x,
            raw_values[:, mode_idx],
            linewidth=2.0,
            linestyle="--",
            color=color,
            label=f"raw $\\sigma_{mode_idx + 1}$",
        )
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Singular value")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, subdir=subdir, stem=stem)
