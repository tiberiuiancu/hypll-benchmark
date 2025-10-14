import torch.nn as nn

from hypll.manifolds import Manifold
import hypll.nn as hnn


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int = 32 * 32 * 3,
        out_size: int = 10,
        hdims: list[int] | None = None,
        manifold: Manifold | None = None,
        activation: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        hdims = hdims or []
        hdims = [in_size] + hdims + [out_size]

        layers = []
        for i in range(len(hdims) - 1):
            layers.append(
                self.make_layer(
                    hdims[i],
                    hdims[i + 1],
                    manifold=manifold,
                    activation=(
                        activation and i < len(hdims) - 2
                    ),  # don't apply relu on the last layer
                )
            )
        self.net = nn.Sequential(*layers)

    def make_layer(
        self,
        in_dim: int,
        out_dim: int,
        manifold: Manifold | None,
        activation: bool = True,
    ):
        if manifold is None:
            layers = [nn.Linear(in_dim, out_dim)] + (
                [] if not activation else [nn.ReLU()]
            )
        else:
            layers = [
                hnn.HLinear(in_dim, out_dim, manifold=manifold),
            ] + (
                []
                if not activation
                else [
                    hnn.HReLU(manifold=manifold),
                ]
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
