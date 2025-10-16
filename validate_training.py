from model_utils import ensure_hypll_repo

ensure_hypll_repo("main")

import torch
import torch.nn as nn

from model_utils import get_dataset

from typing import Literal
from tap import Tap

from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.optim.adam import RiemannianAdam
from hypll.tensors.tangent_tensor import TangentTensor
import hypll.nn as hnn

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000


class Net(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.conv1 = hnn.HConvolution2d(
            in_channels=3, out_channels=6, kernel_size=5, manifold=manifold
        )
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv2 = hnn.HConvolution2d(
            in_channels=6, out_channels=16, kernel_size=5, manifold=manifold
        )
        self.fc1 = hnn.HLinear(
            in_features=16 * 5 * 5, out_features=120, manifold=manifold
        )
        self.fc2 = hnn.HLinear(in_features=120, out_features=84, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=84, out_features=10, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ProfileArgs(Tap):
    dataset: Literal["imagenet", "cifar10", "caltech256"] = "cifar10"
    batch_size: int = 64
    curvature: float = 0.1

    def configure(self):
        self.add_argument("--dataset", "-d")
        self.add_argument("--batch_size", "-b")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    args = ProfileArgs().parse_args()

    trainloader = get_dataset(
        args.dataset,
        args.batch_size,
        flatten=False,
    )

    in0, out0 = next(iter(trainloader))
    in_size = torch.flatten(in0, start_dim=1).shape[-1]
    out_size = out0.shape[-1]

    num_epochs = 5
    results = {}

    # Re-create manifold and model for each config
    manifold1 = PoincareBall(c=Curvature(args.curvature), use_triton_backend=False)
    manifold2 = PoincareBall(c=Curvature(args.curvature), use_triton_backend=True)
    torch.manual_seed(42)
    net1 = Net(manifold1).cuda()
    torch.manual_seed(42)
    net2 = Net(manifold2).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer1 = RiemannianAdam(net1.parameters(), lr=0.001)
    optimizer2 = RiemannianAdam(net2.parameters(), lr=0.001)

    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.cuda(), labels.cuda()

            tangents1 = TangentTensor(data=inputs, man_dim=1, manifold=manifold1)
            inputs1 = manifold1.expmap(tangents1)
            outputs1 = net1(inputs1)
            outputs_tensor1 = outputs1.tensor
            loss1 = criterion(outputs_tensor1, labels)
            optimizer1.zero_grad(set_to_none=True)
            loss1.backward()
            optimizer1.step()

            tangents2 = TangentTensor(data=inputs, man_dim=1, manifold=manifold2)
            inputs2 = manifold2.expmap(tangents2)
            outputs1 = net2(inputs2)
            outputs_tensor2 = outputs1.tensor
            loss2 = criterion(outputs_tensor2, labels)
            optimizer2.zero_grad(set_to_none=True)
            loss2.backward()
            optimizer2.step()

            print(f"{loss2.item() / loss1.item()}")
