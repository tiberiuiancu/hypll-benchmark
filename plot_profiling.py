import json
import pandas as pd


def get_trace_df(path):
    trace_events = json.load(open(path, "r"))["traceEvents"]
    return pd.DataFrame(trace_events)


def get_annotation_interval(d, name, gpu: bool = False):
    cat = ("gpu_" if gpu else "") + "user_annotation"
    dsel = d[(d["cat"] == cat) & (d["name"] == name)].sort_values("ts")
    if dsel.empty:
        raise ValueError(f"annotation '{name}' (gpu={gpu}) not found in trace")
    d = dsel.iloc[-1]
    return d["ts"], d["ts"] + d["dur"]


def filter_df(d, name=None, cat=None, interval=None, pid=None):
    mask = pd.Series(True, index=d.index)
    if name is not None:
        mask &= d["name"] == name
    if cat is not None:
        mask &= d["cat"] == cat
    if interval is not None:
        mask &= (d["ts"] >= interval[0]) & (d["ts"] + d["dur"] <= interval[1])
    if pid is not None:
        mask &= d["pid"] == pid
    return d[mask]


def compute_kernel_breakdown_for_df(df, fwd_interval, bwd_interval):
    def compute_for_interval(interval):
        dfk = filter_df(df, None, "kernel", interval)
        dfa = filter_df(df, None, None, interval, pid=0)
        dfa = dfa[dfa["cat"] != "gpu_user_annotation"]

        def dur_and_count(sub: str, contains: bool = True):
            if contains:
                mask = dfk["name"].str.contains(sub, na=False)
            else:
                mask = dfk["name"].str.startswith(sub, na=False)
            return dfk[mask]["dur"].sum(), int(mask.sum())

        elementwise_d, elementwise_c = dur_and_count("elementwise")
        reduce_d, reduce_c = dur_and_count("reduce")
        gemm_d, gemm_c = dur_and_count("gemm")
        ours_d, ours_c = dur_and_count("_", contains=False)
        other_d = dfa["dur"].sum() - elementwise_d - reduce_d - gemm_d - ours_d
        other_c = len(dfa) - elementwise_c - reduce_c - gemm_c - ours_c
        idle_d = interval[1] - interval[0] - dfa["dur"].sum()
        vals = [gemm_d, reduce_d, elementwise_d, ours_d, other_d, idle_d]
        cnts = [gemm_c, reduce_c, elementwise_c, ours_c, other_c, 0]
        return vals, cnts

    vals_fwd, cnts_fwd = compute_for_interval(fwd_interval)
    vals_bwd, cnts_bwd = compute_for_interval(bwd_interval)
    # return in MICROs (same as trace) along with counts
    return vals_fwd, cnts_fwd, vals_bwd, cnts_bwd


def plot(
    configs,
    title,
    out_path,
    trace_base=".out/traces",
    width=8,
    height_per_config=0.5,
    xlabel="Duration (ms)",
    threshold_ms=0.1,
    # placement
    titlepos=1,
    xlabelpos=0,
    legendpos=-0.5,
    # fonts
    titlesize: int = 10,
    labelsize: int = 10,
    xticksize: int = 9,
    yticksize: int = 9,
    legendsize: int = 9,
):
    """
    Plots grouped by phase: two groups (forward, backward). Inside each group,
    there is one stacked horizontal bar per config (stacked by kernel category).
    Returns path to saved PDF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # categories and colors (kept same order)
    all_labels = ["GEMM", "Reduction", "Elementwise", "Ours", "Other", "Idle"]
    all_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#ff64ef", "#d3d3d3"]

    # load data
    datasets = []
    for cfg in configs:
        path = f"{trace_base}/{cfg}/{cfg}_trace.json"
        df = get_trace_df(path)
        fwd_interval = get_annotation_interval(df, "forward_pass", True)
        try:
            opt_interval = get_annotation_interval(
                df, "Optimizer.step#RiemannianAdam.step", True
            )
        except Exception:
            ann = df[df["cat"] == "gpu_user_annotation"]
            if ann.empty:
                raise ValueError(f"no gpu_user_annotation found for config {cfg}")
            ann = ann.sort_values("ts").iloc[-1]
            opt_interval = (ann["ts"], ann["ts"] + ann["dur"])
        bwd_interval = (fwd_interval[1], opt_interval[0])

        vals_fwd_us, cnts_fwd, vals_bwd_us, cnts_bwd = compute_kernel_breakdown_for_df(
            df, fwd_interval, bwd_interval
        )
        datasets.append(
            {
                "config": cfg,
                "vals_fwd_ms": [v / 1000.0 for v in vals_fwd_us],
                "vals_bwd_ms": [v / 1000.0 for v in vals_bwd_us],
                "cnts_fwd": cnts_fwd,
                "cnts_bwd": cnts_bwd,
            }
        )

    if not datasets:
        raise ValueError("no configs provided")

    # determine which categories to include (non-zero in any config/phase)
    included = [
        any(
            (d["vals_fwd_ms"][i] > threshold_ms) or (d["vals_bwd_ms"][i] > threshold_ms)
            for d in datasets
        )
        for i in range(len(all_labels))
    ]
    idx_map = [i for i, inc in enumerate(included) if inc]
    filtered_labels = [l for l, inc in zip(all_labels, included) if inc]
    filtered_colors = [c for c, inc in zip(all_colors, included) if inc]

    # compute x limit
    max_total = 0.0
    for d in datasets:
        max_total = max(
            max_total,
            sum(d["vals_fwd_ms"][i] for i in idx_map),
            sum(d["vals_bwd_ms"][i] for i in idx_map),
        )
    xlim = max(max_total + 1, 1.0)

    # layout y positions: group forward first, then backward. Within each group, configs stacked vertically.
    n_cfg = len(datasets)
    group_gap = 0.8
    bar_height = 0.8
    # create y positions
    y_positions = {"fwd": [], "bwd": []}
    total_rows = n_cfg * 2 + 1  # +1 gap
    start_y = total_rows / 2.0
    # forward group: y = start_y, start_y-1, ...
    cur = start_y
    for i in range(n_cfg):
        y_positions["fwd"].append(cur)
        cur -= 1.0
    cur -= group_gap
    for i in range(n_cfg):
        y_positions["bwd"].append(cur)
        cur -= 1.0

    # plotting
    height = max(2.0, height_per_config * n_cfg * 2)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0.0, xlim)
    ax.set_ylim(cur + 0.5, start_y + 0.5)

    # helper: draw stacked bar at y for list of segment lengths (ms) with colors
    def _draw_stacked(ax, y, segs, counts, colors):
        left = 0.0
        for seg_len, cnt, color in zip(segs, counts, colors):
            if seg_len <= threshold_ms:
                left += 0.0
                continue
            ax.barh([y], [seg_len], left=[left], height=bar_height, color=color)[0]
            # annotation: place count centered in segment if there's room; choose color for contrast
            annot_color = "white"
            if cnt > 0:
                ax.text(
                    left + seg_len / 2,
                    y,
                    str(int(cnt)),
                    va="center",
                    ha="center",
                    color=annot_color,
                    fontsize=8,
                )
            left += seg_len

    # draw all configs: in each group iterate configs in same order as provided
    for idx, d in enumerate(datasets):
        # forward
        y_f = y_positions["fwd"][idx]
        segs_f = [d["vals_fwd_ms"][i] for i in idx_map]
        cnts_f = [d["cnts_fwd"][i] for i in idx_map]
        _draw_stacked(ax, y_f, segs_f, cnts_f, filtered_colors)
        # backward
        y_b = y_positions["bwd"][idx]
        segs_b = [d["vals_bwd_ms"][i] for i in idx_map]
        cnts_b = [d["cnts_bwd"][i] for i in idx_map]
        _draw_stacked(ax, y_b, segs_b, cnts_b, filtered_colors)

    # y ticks: place one tick per bar (per config, per group)
    def _map_cfg_name(cfg):
        if cfg.startswith("h_t_mlp"):
            return "Ours"
        elif cfg.startswith("h_mlp"):
            return "HypLL"
        else:
            return "Euclidean"

    is_large_plot = width > 5
    y_tick_positions = y_positions["fwd"] + y_positions["bwd"]
    if not is_large_plot and len(configs) == 1:
        y_tick_labels = ["fwd", "bwd"]
    else:
        y_tick_labels = [f"{_map_cfg_name(cfg)} (fwd)" for cfg in configs] + [
            f"{_map_cfg_name(cfg)} (bwd)" for cfg in configs
        ]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=yticksize, rotation=45)

    ax.tick_params(axis="x", labelsize=xticksize)
    if title:
        fig.suptitle(title, fontsize=titlesize, y=titlepos)
    if xlabel:
        fig.supxlabel(xlabel, fontsize=labelsize, y=xlabelpos)

    # legend
    handles = [
        Patch(facecolor=c, label=l) for l, c in zip(filtered_labels, filtered_colors)
    ]
    ncol = len(handles) if is_large_plot else width // 1.5
    ax.legend(
        handles=handles,
        fontsize=legendsize,
        loc="lower center",
        bbox_to_anchor=(0.5, legendpos),
        ncol=ncol,
        frameon=False,
    )

    fig.subplots_adjust(left=0.12, right=0.82, top=0.92, bottom=0.18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


plot(
    ["h_mlp_main"],
    title="HypLL FC Layer Kernel Duration",
    out_path=".out/plots/kernels_hypll.pdf",
    legendpos=-0.6,
)

plot(
    ["h_mlp_main", "h_t_mlp_main", "mlp_main"],
    title="GPU Kernel Durations grouped by forward / backward",
    out_path=".out/plots/kernels_grouped.pdf",
    height_per_config=0.3,
    legendpos=-0.45,
    xlabelpos=-0.025,
)
