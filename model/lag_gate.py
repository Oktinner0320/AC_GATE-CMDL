import copy
import math
from typing import List, Dict, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F



class GatedLinearUnit(nn.Module):
    """
    This module is also known as  **GLU** - Formulated in:
    `Dauphin, Yann N., et al. "Language modeling with gated convolutional networks."
    International conference on machine learning. PMLR, 2017
    <https://arxiv.org/abs/1612.08083>`_.

    The output of the layer is a linear projection (X * W + b) modulated by the gates **sigmoid** (X * V + c).
    These gates multiply each element of the matrix X * W + b and control the information passed on in the hierarchy.
    This unit is a simplified gating mechanism for non-deterministic gates that reduce the vanishing gradient problem,
    by having linear units coupled to the gates. This retains the non-linear capabilities of the layer while allowing
    the gradient to propagate through the linear unit without scaling.

    Parameters
    ----------
    input_dim: int
        The embedding size of the input.
    """

    def __init__(self, input_dim: int):
        super(GatedLinearUnit, self).__init__()

        # Two dimension-preserving dense layers
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)
    

class GatedResidualNetwork(nn.Module):
    """
    This module, known as **GRN**, takes in a primary input (x) and an optional context vector (c).
    It uses a ``GatedLinearUnit`` for controlling the extent to which the module will contribute to the original input
    (x), potentially skipping over the layer entirely as the GLU outputs could be all close to zero, by that suppressing
    the non-linear contribution.
    In cases where no context vector is used, the GRN simply treats the context input as zero.
    During training, dropout is applied before the gating layer.

    Parameters
    ----------
    input_dim: int
        The embedding width/dimension of the input.
    hidden_dim: int
        The intermediate embedding width.
    output_dim: int
        The embedding width of the output tensors.
    dropout: Optional[float]
        The dropout rate associated with the component.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                dropout: Optional[float] = 0.05,
                context_dim: Optional[int] = None,
                batch_first: Optional[bool] = True):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # =================================================
        # Input conditioning components (Eq.4 in the original paper)
        # =================================================
        # for using direct residual connection the dimension of the input must match the output dimension.
        # otherwise, we'll need to project the input for creating this residual connection
        self.project_residual: bool = self.input_dim != self.output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_dim, self.output_dim))

        # A linear layer for projecting the primary input (acts across time if necessary)
        self.fc1 = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim), batch_first=batch_first)

        # In case we expect context input, an additional linear layer will project the context
        if self.context_dim is not None:
            self.context_projection = TimeDistributed(nn.Linear(self.context_dim, self.hidden_dim, bias=False),
                                                    batch_first=batch_first)
        # non-linearity to be applied on the sum of the projections
        self.elu1 = nn.ELU()

        # ============================================================
        # Further projection components (Eq.3 in the original paper)
        # ============================================================
        # additional projection on top of the non-linearity
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_dim, self.output_dim), batch_first=batch_first)

        # ============================================================
        # Output gating components (Eq.2 in the original paper)
        # ============================================================
        self.dropout = nn.Dropout(self.dropout)
        self.gate = TimeDistributed(GatedLinearUnit(self.output_dim), batch_first=batch_first)
        self.layernorm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)

    def forward(self, x, context=None):

        # compute residual (for skipping) if necessary
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x
        # ===========================
        # Compute Eq.4
        # ===========================
        x = self.fc1(x)
        if context is not None:
            context = self.context_projection(context)
            x = x + context

        # compute eta_2 (according to paper)
        x = self.elu1(x)

        # ===========================
        # Compute Eq.3
        # ===========================
        # compute eta_1 (according to paper)
        x = self.fc2(x)

        # ===========================
        # Compute Eq.2
        # ===========================
        x = self.dropout(x)
        x = self.gate(x)
        # perform skipping using the residual
        x = x + residual
        # apply normalization layer
        x = self.layernorm(x)

        return x