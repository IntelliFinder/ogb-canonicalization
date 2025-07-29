import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

from data.models.layers.egnn_layer import EGNNLayer


class EGNNModel(torch.nn.Module):
    """
    E-GNN model from "E(n) Equivariant Graph Neural Networks".
    """
    def __init__(
        self,
        num_layers: int = 5,
        emb_dim: int = 128,
        proj_dim: int = 10,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "add",
        pool: str = "add",
        residual: bool = True
    ):
        """
        Initializes an instance of the EGNNModel class with the provided parameters.

        Parameters:
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Dimension of the node embeddings (default: 128)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - activation (str): Activation function to be used (default: "relu")
        - norm (str): Normalization method to be used (default: "layer")
        - aggr (str): Aggregation method to be used (default: "add")
        - pool (str): Global pooling method to be used (default: "add")
        - residual (bool): Whether to use residual connections (default: True)
        """
        super().__init__()
        self.residual = residual

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, proj_dim, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "add": global_add_pool}[pool]

    def forward(self, batch):
        batch = batch.to(self.convs[0].mlp_msg[0].weight.device)
        h = batch.x.squeeze(1).to(batch.pos.device)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            #print(batch.edge_index[:3])
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update
    
        return pos, h  # (batch_size, out_dim)