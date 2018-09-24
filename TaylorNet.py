import torch
import torch.nn as nn
import math

class TaylorNet(nn.Module):
    r"""Applies a non-linear multiplicative transformation to the incoming data,
    in order to generate output features that can be quadratic and linear in the 
    input features: 
    :math:`y = (x W_2^T) * (x W_1^T) + x W_1^T + b`
    
    Note that if output size = input size, then W_2 is not used, and the
    transformation becomes:
    :math:`y = x * (x W^T) + x W^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where :math:`*` means any number of
          additional dimensions

    Attributes:
        weight_1: the learnable weights of the module of shape
            `(out_features x in_features)`
        weight_2: the learnable weights of the module of shape
            `(out_features x in_features)`
            If out_features = in_features, there is no weight_2 matrix
        bias:   the learnable bias of the module of shape `(in_features)`

    Examples::

        >>> m = nn.TaylorNet(5)
        >>> input = torch.randn(128, 5)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features=None, bias=True):
        super(TaylorNet, self).__init__()
        if out_features is None:
            out_features = in_features
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        if (in_features != out_features):
            self.weight2 = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.weight2 = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        nn.init.xavier_normal_(self.weight1)
        if self.weight2 is not None:
            nn.init.xavier_normal_(self.weight2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        Wx = input.matmul(self.weight1.t())
        x = input
        if self.weight2 is not None:
            x = input.matmul(self.weight2.t())
        output = x.mul(Wx) + Wx
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )