import pandas as pd
import json
import matplotlib.pyplot as plt


def get_trace_df(path):
    trace_events = json.load(open(path, "r"))["traceEvents"]
    return pd.DataFrame(trace_events)


def get_first_annotation_interval(d, name, gpu: bool = False):
    cat = ("gpu_" if gpu else "") + "user_annotation"
    d = d[(d["cat"] == cat) & (d["name"] == name)].sort_values("ts").iloc[0]
    start = d["ts"]
    end = start + d["dur"]
    return start, end


def filter_df(d, name, cat, interval):
    mask = d.index == d.index
    if name is not None:
        mask &= d["name"] == name
    if cat is not None:
        mask &= d["cat"] == cat
    if interval is not None:
        mask &= (d["ts"] >= interval[0]) & (d["ts"] + d["dur"] <= interval[1])
    return d[mask]


def kernel_duration_by_name(d, name):
    return d[d["name"].str.contains(name)]["dur"].sum()


df = get_trace_df(".out/traces/h_mlp_main/h_mlp_main_trace.json")
fwd_interval = get_first_annotation_interval(df, "forward_pass", True)
opt_interval = get_first_annotation_interval(
    df, "Optimizer.step#RiemannianAdam.step", True
)
bwd_interval = (fwd_interval[1], opt_interval[0])


def compute_kernel_breakdown(interval):
    dfk = filter_df(df, None, "kernel", interval)

    def dur_and_count(sub):
        mask = dfk["name"].str.contains(sub)
        return dfk[mask]["dur"].sum(), int(mask.sum())

    elementwise_d, elementwise_c = dur_and_count("elementwise")
    reduce_d, reduce_c = dur_and_count("reduce")
    gemm_d, gemm_c = dur_and_count("gemm")
    idle_d = interval[1] - interval[0] - elementwise_d - reduce_d - gemm_d
    return [elementwise_d, reduce_d, gemm_d, idle_d], [
        elementwise_c,
        reduce_c,
        gemm_c,
        0,
    ]


def plot_kernel_breakdown(
    width=8,
    height=4,
    title_size=14,
    axis_label_size=12,
    tick_label_size=10,
    legend_size=10,
    annotation_size=9,
    split_at=1.0,  # if None -> no split; else split here
    right_max=15.0,  # max x on the right pane (or overall if no split)
    ylabel="",
    title="Kernel Durations Per Training Step",
    xlabel="Duration (ms)",
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    labels = ["Element-\nwise", "Reduction", "GEMM", "Idle"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d3d3d3"]
    positions = [1, 0]
    bar_height = 0.35

    # ---- data ----
    vals_fwd, cnts_fwd = compute_kernel_breakdown(fwd_interval)
    vals_bwd, cnts_bwd = compute_kernel_breakdown(bwd_interval)
    vals_fwd = [v / 1000.0 for v in vals_fwd]  # µs -> ms
    vals_bwd = [v / 1000.0 for v in vals_bwd]
    totals = [sum(vals_fwd), sum(vals_bwd)]
    overall_max = max(totals + [float(right_max)])

    # ---- helpers ----
    def _bar(ax, y, left, length, color):
        if length <= 0:
            return None
        return ax.barh([y], [length], left=[left], height=bar_height, color=color)[0]

    def _annot(ax, x_center, y, txt, bg_color):
        if txt <= 0:
            return
        ax.text(
            x_center,
            y,
            str(txt),
            ha="center",
            va="center",
            color=("black" if bg_color == "#d3d3d3" else "white"),
            fontsize=annotation_size,
            rotation=0,
        )

    # ---- figure/axes ----
    if split_at is None:
        # Single axis
        fig, ax = plt.subplots(figsize=(width, height))
        lefts = [0.0, 0.0]
        for idx, (lbl, color) in enumerate(zip(labels, colors)):
            for row, seg_len in enumerate([vals_fwd[idx], vals_bwd[idx]]):
                L, R = lefts[row], lefts[row] + seg_len
                _bar(ax, positions[row], L, seg_len, color)
                # annotate at segment center
                count = (cnts_fwd if row == 0 else cnts_bwd)[idx]
                _annot(ax, L + seg_len / 2, positions[row], count, color)
                lefts[row] = R

        ax.set_xlim(0.0, overall_max)
        ax.set_yticks(positions)
        ax.set_yticklabels(["forward", "backward"], fontsize=tick_label_size)
        ax.tick_params(axis="x", labelsize=tick_label_size)

        if title:
            fig.suptitle(title, fontsize=title_size, y=1.02)
        if xlabel:
            fig.supxlabel(xlabel, fontsize=axis_label_size, y=0.02)
        if ylabel:
            fig.supylabel(ylabel, fontsize=axis_label_size, x=0.06)

        handles = [Patch(facecolor=c, label=l) for l, c in zip(labels, colors)]
        fig.legend(
            handles=handles,
            fontsize=legend_size,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(handles),
            frameon=False,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.18)

    else:
        # Split axis
        split_at = max(0.0, min(float(split_at), float(right_max) - 1e-6))
        left_xlim = (0.0, split_at)
        right_xlim = (split_at, float(right_max))

        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(1, 2, width_ratios=[0.8, 0.2], wspace=0.0)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1], sharey=axL)

        axL.spines["right"].set_visible(False)
        axR.spines["left"].set_visible(False)
        axL.tick_params(right=False, left=True, labelleft=True)
        axR.tick_params(left=False, labelleft=False)
        axR.set_facecolor(axL.get_facecolor())

        axL.set_xlim(*left_xlim)
        axR.set_xlim(*right_xlim)

        def _intersect(a0, a1, b0, b1):
            L = max(a0, b0)
            R = min(a1, b1)
            return L, max(0.0, R - L)

        lefts = [0.0, 0.0]
        for idx, (lbl, color) in enumerate(zip(labels, colors)):
            for row, seg_len in enumerate([vals_fwd[idx], vals_bwd[idx]]):
                L = lefts[row]
                R = L + seg_len
                lL, lLen = _intersect(L, R, *left_xlim)
                rL, rLen = _intersect(L, R, *right_xlim)

                _bar(axL, positions[row], lL, lLen, color)
                _bar(axR, positions[row], rL, rLen, color)

                count = (cnts_fwd if row == 0 else cnts_bwd)[idx]
                if lLen > 0:
                    _annot(axL, lL + lLen / 2, positions[row], count, color)
                elif rLen > 0:
                    _annot(axR, rL + rLen / 2, positions[row], count, color)

                lefts[row] = R

        axL.set_yticks(positions)
        axL.set_yticklabels(["fwd", "bwd"], fontsize=tick_label_size)
        axL.tick_params(axis="x", labelsize=tick_label_size)
        axR.tick_params(axis="x", labelsize=tick_label_size)

        if title:
            fig.suptitle(title, fontsize=title_size, y=1.1)
        if xlabel:
            fig.supxlabel(xlabel, fontsize=axis_label_size, y=0.02)
        if ylabel:
            fig.supylabel(ylabel, fontsize=axis_label_size, x=0.06)

        handles = [Patch(facecolor=c, label=l) for l, c in zip(labels, colors)]
        fig.legend(
            handles=handles,
            fontsize=legend_size,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.85),
            ncol=len(handles),
            frameon=False,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.18, wspace=0.0)

    fig.savefig(".out/plots/kernels_fwd_bwd.pdf", bbox_inches="tight")
    plt.close(fig)


plot_kernel_breakdown(
    width=3.4,  # inches, fits a single column (ICML column width ≈ 3.4in)
    height=2.0,  # compact height for figure
    title_size=10,  # match main text size
    axis_label_size=9,
    tick_label_size=8,
    legend_size=8,
    annotation_size=7,
    split_at=None,
    right_max=9,
)
