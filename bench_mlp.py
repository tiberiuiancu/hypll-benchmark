from utils import ensure_hypll_repo

import os
import argparse

ref = os.getenv("REF", "main")
ensure_hypll_repo(ref)

from models.mlp import MLP

import torch
import triton

from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.tensors.tangent_tensor import TangentTensor

# configs for B, M, K
default_b, default_m, default_k, default_d = 256, 4096, 4096, 3
activation = False

sweeps = {
    "b": [(2**i, default_k, default_m, default_d) for i in range(12)],
    "k": [(default_b, 2**i, default_m, 1) for i in range(7, 15)],
    "m": [(default_b, 2**i, 2**i, default_d) for i in range(7, 14)],
    "d": [(default_b, default_k, default_m, i) for i in range(1, 9)],
}


configs = {
    "triton": {
        "line_name": f"Triton {ref}",
        "style": ("red", "-"),
    },
    "torch": {
        "line_name": "PyTorch",
        "style": ("blue", "--"),
    },
    "compile": {
        "line_name": "Compiled PyTorch",
        "style": ("green", "-."),
    },
    "euclidean": {
        "line_name": "Euclidean",
        "style": ("orange", ":"),
    },
}


def get_active_configs():
    return os.getenv("BENCH_CONFIGS", "torch triton euclidean").split(" ")


active_configs = get_active_configs()


def get_bench_kwargs():
    return dict(
        line_arg="provider",
        line_vals=active_configs,
        line_names=[configs[config]["line_name"] for config in active_configs],
        styles=[configs[config]["style"] for config in active_configs],
        ylabel="Execution Time (ms)",
        args={"c": 0.1},
        x_log=True,
    )


def build_model(
    k,
    m,
    d,
    c,
    dtype,
    device,
    hyperbolic: bool = False,
    use_triton_backend: bool = False,
    compiled: bool = False,
):
    manifold = None
    if hyperbolic:
        manifold = PoincareBall(Curvature(c), use_triton_backend=use_triton_backend)

    hdims = [m] * (int(d) - 1)
    model = MLP(k, m, hdims, manifold, activation=activation).to(device, dtype)

    if compiled:
        model.compile()

    return model, manifold


def bench(b, k, m, d, c, provider):
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    compiled, use_triton_backend, hyperbolic = {
        "triton": (False, True, True),
        "torch": (False, False, True),
        "compile": (True, False, True),
        "euclidean": (False, False, False),
    }[provider]

    model, manifold = build_model(
        k,
        m,
        d,
        c,
        dtype=torch.float32,
        device="cuda",
        hyperbolic=hyperbolic,
        use_triton_backend=use_triton_backend,
        compiled=compiled,
    )
    model.train()

    def run():
        x = torch.randn(b, k, device=device, dtype=dtype)
        y = torch.randn(b, m, device=device, dtype=dtype)
        if provider != "euclidean":
            tangents = TangentTensor(data=x, manifold=manifold)
            x = manifold.expmap(tangents)
        y_hat = model(x)
        if provider != "euclidean":
            y_hat = y_hat.tensor
        torch.nn.functional.cross_entropy(y_hat, y).backward()
        torch.cuda.synchronize()

    ms = triton.testing.do_bench(run)
    return ms


def get_bench(dim: str):
    decorator = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_vals=sweeps[dim],
            x_names=["b", "k", "m", "d"],
            plot_name=dim,
            **get_bench_kwargs(),
        )
    )

    return decorator(bench)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Poincare FC Kernel")
    parser.add_argument(
        "--activation", "-a", action="store_true", help="Use activation function"
    )
    args = parser.parse_args()
    activation = args.activation

    # Run benchmarks and save output
    save_path = f".out/bench/FC_bench_{ref}"
    os.makedirs(save_path, exist_ok=True)
    for dim in sweeps.keys():
        get_bench(dim).run(
            show_plots=False,
            print_data=True,
            save_path=save_path,
        )
