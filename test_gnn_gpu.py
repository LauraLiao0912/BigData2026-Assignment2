
import torch
from torch_geometric.nn import SAGEConv

assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device("cuda")

# 4 nodes, each with 8 input features
x = torch.randn(4, 8, device=device)

# Directed edges: source -> target
edge_index = torch.tensor(
    [[0, 1, 2, 3, 0, 2],
     [1, 2, 3, 0, 2, 1]],
    dtype=torch.long,
    device=device
)

# One GraphSAGE message-passing layer
model = SAGEConv(
    in_channels=8,
    out_channels=16
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

optimizer.zero_grad()

output = model(x, edge_index)

loss = output.square().mean()
loss.backward()

optimizer.step()

torch.cuda.synchronize()

print("Device:", output.device)
print("Output shape:", tuple(output.shape))
print("Loss:", round(loss.item(), 6))
print("GNN GPU test passed")
