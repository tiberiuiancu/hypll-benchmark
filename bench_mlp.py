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
default_b, default_m, default_k, default_d = 256, 4096, 4096, 2
activation = False
b_sweep = [2**i for i in range(12)]
m_sweep = [2**i for i in range(7, 15)]
k_sweep = [2**i for i in range(7, 15)]
d_sweep = [i for i in range(10)]

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
    M,
    K,
    D,
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

    hdims = [M] * (int(D) - 1)
    model = MLP(K, M, hdims, manifold, activation=activation).to(device, dtype)

    if compiled:
        model.compile()

    return model, manifold


def bench(B, M, K, D, c, provider):
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
        M,
        K,
        D,
        c,
        dtype=torch.float32,
        device="cuda",
        hyperbolic=hyperbolic,
        use_triton_backend=use_triton_backend,
        compiled=compiled,
    )
    model.train()

    def run():
        x = torch.randn(B, K, device=device, dtype=dtype)
        y = torch.randn(B, M, device=device, dtype=dtype)
        if provider != "euclidean":
            tangents = TangentTensor(data=x, manifold=manifold)
            x = manifold.expmap(tangents)
        y_hat = model(x)
        if provider != "euclidean":
            y_hat = y_hat.tensor
        torch.nn.functional.cross_entropy(y_hat, y).backward()

    ms = triton.testing.do_bench(run)
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=b_sweep,
        plot_name=f"poincare_fc_performance_b",
        **get_bench_kwargs(),
    )
)
def bench_b(batch_size, provider: str, c: float):
    return bench(batch_size, default_m, default_k, default_d, c, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["input_size"],
        x_vals=m_sweep,
        plot_name=f"poincare_fc_performance_m",
        **get_bench_kwargs(),
    )
)
def bench_m(input_size, provider: str, c: float):
    return bench(default_b, input_size, default_k, default_d, c, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size"],
        x_vals=k_sweep,
        plot_name=f"poincare_fc_performance_k",
        **get_bench_kwargs(),
    )
)
def bench_k(hidden_size, provider: str, c: float):
    return bench(default_b, default_m, hidden_size, default_d, c, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["depth"],
        x_vals=d_sweep,
        plot_name=f"poincare_fc_performance_d",
        **(get_bench_kwargs() | {"x_log": False}),
    )
)
def bench_d(depth, provider: str, c: float):
    return bench(default_b, default_m, default_k, depth, c, provider)


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
    run_kwargs = dict(show_plots=False, print_data=True, save_path=save_path)
    bench_d.run(**run_kwargs)
    bench_m.run(**run_kwargs)
    bench_b.run(**run_kwargs)
    bench_k.run(**run_kwargs)
