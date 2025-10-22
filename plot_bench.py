import os
import pandas as pd


refs = [
    "triton",
    "memory",
    # "logmap0",
    # "expmap0",
    "fused-relu-logmap0",
    "fused-op-in-tangent-space",
    "fc-fwd-1d-grid",
    "fc-bwd-1d-grid",
]

dims = ["d", "b", "m", "k"]
dfs = {}

for dim in dims:
    dfs_dim = []
    for ref in refs:
        dfs_dim.append(
            pd.read_csv(f".out/bench/FC_bench_{ref}/poincare_fc_performance_{dim}.csv")
        )
    df_concat = pd.concat(dfs_dim, axis=1)
    df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]
    dfs[dim] = df_concat
    print(dfs[dim])


for dim in dims:
    df = dfs[dim]
    if dim != "d":
        ax = df.plot(x=0, logx=True)
    else:
        ax = df.plot(x=0)
    fig = ax.get_figure()
    os.makedirs(".out/plots", exist_ok=True)
    fig.savefig(f".out/plots/{dim}.png")
    fig.clf()
