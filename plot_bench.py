import os
import pandas as pd
import matplotlib.pyplot as plt


def ms_to_tflops(ms, b, k, m, d, hyperbolic: bool = True):
    def flops_per_hyp_layer(b, k, m):
        fc_fwd_kernel = b * (2 * k + 29 * m + 12)
        fc_fwd_matmul = 2 * b * m * k
        fc_fwd_norm = 2 * k * m
        fc_fwd = fc_fwd_kernel + fc_fwd_matmul + fc_fwd_norm

        fc_bwd_kernel = 84 * b * m
        fc_bwd_matmul = 4 * b * m * k
        fc_bwd_vecmul = 2 * b * k + 2 * k * m
        fc_bwd = fc_bwd_kernel + fc_bwd_matmul + fc_bwd_vecmul

        logmap_fwd = 4 * b * m
        relu_fwd = b * m
        expmap_fwd = 7 * b * m
        activation_fwd = logmap_fwd + relu_fwd + expmap_fwd

        logmap_bwd = 9 * b * m
        expmap_bwd = 5 * b * m
        activation_bwd = logmap_bwd + expmap_bwd

        fc = fc_fwd + fc_bwd
        activation = activation_fwd + activation_bwd

        return fc + activation

    def flops_per_euc_layer(b, k, m):
        return 6 * b * k * m + 2 * b * m

    flops_per_layer = flops_per_hyp_layer if hyperbolic else flops_per_euc_layer
    total_flops = flops_per_layer(b, k, m) + flops_per_layer(b, m, m) * (d - 1)
    total_tflops = total_flops / 1e12
    total_seconds = ms / 1e3
    total_tflops_second = total_tflops / total_seconds
    return total_tflops_second


refs = [
    "triton",
    "memory",
    # "logmap0",
    # "expmap0",
    "fused-relu-logmap0",
    # "fused-op-in-tangent-space",
    # "fc-fwd-1d-grid",
    "fc-bwd-1d-grid",
]

dims = ["d", "b", "m", "k"]
dfs = {}

for dim in dims:
    dfs_dim = []
    for ref in refs:
        df = pd.read_csv(f".out/bench/FC_bench_{ref}/{dim}.csv")
        df = df.astype(float)
        df.rename(
            columns={
                "PyTorch": "HypLL",
                "Triton fused-relu-logmap0": "Separate fused",
                "Triton fc-bwd-1d-grid": "Ours",
                "Triton triton": "Fused FC only",
                "Triton memory": "Fused FC only",
                "Euclidean": "Euclidean",
            },
            inplace=True,
        )
        dfs_dim.append(df)

    df_concat = pd.concat(dfs_dim, axis=1)
    df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]
    dfs[dim] = df_concat


def plot_dims(
    title_size: int = 14,
    font_size: int = 10,
    fig_width: int = 10,
    fig_height: int = 6,
    plot_flops: bool = False,
    output_path: str = ".out/plots/combined.pdf",
):
    plt.rcParams.update({"font.size": font_size})
    n = len(dims)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # Define custom styles: markers and colors
    markers = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">"]
    colors = plt.get_cmap("tab10").colors  # 10 distinct colors

    handles, labels = None, None

    for i, dim in enumerate(dims):
        df = dfs[dim]

        ax = axes[i]
        # Get columns to plot (skip the dimension columns)
        b, k, m, d = [df[c] for c in ["b", "k", "m", "d"]]
        columns = df.columns[4:]
        x = df[dim]
        lines = []
        for idx, col in enumerate(columns):
            y = df[col]
            if plot_flops:
                y = ms_to_tflops(y, b, k, m, d, hyperbolic=(col != "Euclidean"))
            (line,) = ax.plot(
                x,
                y,
                label=col,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=6,
            )
            lines.append(line)
        if dim != "d":
            ax.set_xscale("log", base=2)
        ax.set_title(f"sweep {dim}", fontsize=title_size)
        if i % ncols == 0:
            if plot_flops:
                ax.set_ylabel("TFLOP/s")
            else:
                ax.set_ylabel("Duration (ms)")
        ax.grid(True)  # Turn on grid
        if plot_flops:
            ax.axhline(y=7.465, color="gray", linestyle=":", linewidth=1)
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared legend below all subplots
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=(len(labels) + 1) // 2,  # split legend into 2 rows
        bbox_to_anchor=(0.5, -0.08),
        fontsize=font_size,
        frameon=False,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


plot_dims(title_size=14, font_size=10, fig_width=6.75, fig_height=5, plot_flops=True)
