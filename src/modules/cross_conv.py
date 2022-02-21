import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from inspect import Parameter
from torch_sparse import SparseTensor


class CrossConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (float, optional): If set, will combine aggregation and
            skip information via :math:`\beta\,\mathbf{W}_1 \vec{x}_i + (1 -
            \beta) \left(\sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2
            \vec{x}_j \right)`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 beta: Optional[float] = None, dropout: float = 0.,
                 edge_dim: Optional[int] = None, bias: bool = True, **kwargs):
        super(CrossConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_title = Linear(in_channels[0], heads * out_channels)
        self.lin_ent = Linear(in_channels[1], heads * out_channels)
        self.lin_value_title = Linear(in_channels[0], heads * out_channels)
        self.lin_value_ent = Linear(in_channels[1], heads * out_channels)
        self.lin_out_title = Linear(heads * out_channels, heads * out_channels)
        self.lin_out_ent = Linear(heads * out_channels, heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = Linear(1, 1)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_title.reset_parameters()
        self.lin_ent.reset_parameters()
        self.lin_value_title.reset_parameters()
        self.lin_value_ent.reset_parameters()
        self.lin_out_title.reset_parameters()
        self.lin_out_ent.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index,
                                         j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        elif isinstance(edge_index, SparseTensor):
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_weight'] = edge_index.storage.value()
            out['edge_attr'] = edge_index.storage.value()
            out['edge_type'] = edge_index.storage.value()

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']
        out['direction'] = kwargs.get('direction', 0)

        return out

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, direction=0):

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None, direction=0)

        if self.concat:
            if direction == 0:
                out = self.lin_out_ent(out.view(-1, self.heads * self.out_channels))
            else:
                out = self.lin_out_title(out.view(-1, self.heads * self.out_channels))
        else:
            out = out.mean(dim=1)

        # x = self.lin_skip(x[1])
        # if self.beta is not None:
        #     out = self.beta * x + (1 - self.beta) * out
        # else:
        #     out += x

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor],
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], direction: int) -> Tensor:
        if direction == 0:
            # source entity => target title
            # query = title
            # key = entity
            query = self.lin_title(x_j).view(-1, self.heads, self.out_channels)
            key = self.lin_ent(x_i).view(-1, self.heads, self.out_channels)
        else:
            query = self.lin_ent(x_j).view(-1, self.heads, self.out_channels)
            key = self.lin_title(x_i).view(-1, self.heads, self.out_channels)

        lin_edge = self.lin_edge
        if edge_attr is not None:
            edge_attr = lin_edge(edge_attr).view(-1, self.heads,
                                                 self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if direction == 0:
            out = self.lin_value_ent(x_j).view(-1, self.heads, self.out_channels)
        else:
            out = self.lin_value_title(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)