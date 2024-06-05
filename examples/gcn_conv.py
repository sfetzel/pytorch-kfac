from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul

class WeightedMessagePassing(MessagePassing):
    """
    Message passing which also considers the edge weights.
    """

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index` and applies multiplies the messages with the
        corresponding edge weight.
        """
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`
        or a :obj:`torch.sparse.Tensor`.
        """
        return matmul(adj_t, x, reduce=self.aggr)


class GCNConv(WeightedMessagePassing):
    """
    A GCN Convolution layer which first executes message passing,
    then applies a linear transform.
    """

    def __init__(self, d_in, d_out, cached=True, bias=True):
        super(GCNConv, self).__init__()
        self.lin = Linear(d_in, d_out, bias=bias)
        self.bias = self.lin.bias
        #self.message_passing = WeightedMessagePassing()
        self._cached_edge_index = None

    def reset_parameters(self):
        glorot(self.lin.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_index):
        if self._cached_edge_index is None:
            self._cached_edge_index = gcn_norm(  # yapf: disable
                edge_index, None, x.size(self.node_dim),
                False, True, dtype=x.dtype)
        edge_index_normalized, edge_weight_normalized = self._cached_edge_index

        x = self.propagate(edge_index_normalized, x=x,
                                             edge_weight=edge_weight_normalized)

        x = self.lin(x)
        return x
