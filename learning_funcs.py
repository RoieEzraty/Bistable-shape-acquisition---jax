from __future__ import annotations

import copy
import numpy as np

from typing import Tuple, List
from numpy import array, zeros
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Union, Optional

if TYPE_CHECKING:
    from StructureClass import StructureClass
    from VariablesClass import VariablesClass
    from EqClass import EqClass


# ===================================================
# file_funcs - functions to assist with file conversions etc.
# ===================================================


def build_incidence(Nin: int, Nout: int) -> Tuple[NDArray[np.int_], NDArray[np.int_], List[NDArray[np.int_]],
                                                  NDArray[np.int_], int, int]:
    """
    Builds incidence matrix DM as np.array [NEdges, NNodes] for 1 single FC network, w/out ground
    its meaning is 1 at input node and -1 at outpus for every row which resembles one edge.

    input (extracted from Variabs input):
    Strctr: "Network_Structure" class instance with the input, intermediate and output nodes

    output:
    EI, EJ     - 1D np.arrays sized NEdges such that EI[i] is node connected to EJ[i] at certain edge
    EIEJ_plots - EI, EJ divided to pairs for ease of use
    DM         - Incidence matrix as np.array [NEdges, NNodes]
    NE         - NEdges, int
    NN         - NNodes, int
    """
    input_nodes_arr = np.arange(Nin)
    output_nodes_arr = np.arange(Nout)+Nin

    NN: int = Nin + Nout
    EIlst: List[int] = []
    EJlst: List[int] = []

    for inNode in input_nodes_arr:
        for outNode in output_nodes_arr:
            EIlst.append(inNode)
            EJlst.append(outNode)

    EI: NDArray[np.int_] = array(EIlst)
    EJ: NDArray[np.int_] = array(EJlst)
    NE: int = len(EI)

    # for plots
    EIEJ_plots: List = [(EI[i], EJ[i]) for i in range(len(EI))]

    DM: NDArray[np.int_] = zeros([NE, NN], dtype=np.int_)  # Incidence matrix
    for i in range(NE):
        DM[i, int(EI[i])] = +1.
        DM[i, int(EJ[i])] = -1.

    return EI, EJ, EIEJ_plots, DM, NE, NN, output_nodes_arr


def inverse_incidence(DM: NDArray[np.int_]) -> NDArray[np.float_]:
    """
    inverts incidence matrix, should be done once for GD-like scheme

    input:
    DM - Incidence matrix np.array [NE, NN]

    output:
    DM_dagger - Shortened Lagrangian np.array cubic array sized [NNodes]
    """
    return np.linalg.pinv(DM)


def grad_loss_FC(NE: int, inputs_normalized: NDArray[np.float_], outputs_normalized: NDArray[np.int_], DM: NDArray[np.int_],
                 output_nodes_arr: NDArray[np.int_], loss: NDArray[np.float_]) -> NDArray[np.float_]:

    """
    Compute the gradient of the loss function with respect to the edge pressures in a fully connected network.
    As in appendix "Comparison to gradient descent" in the paper.

    Parameters:
    - NE: int
        Number of edges in the network.
    - p: NDArray[np.float_]
        Array of node pressures.
    - DM: NDArray[np.int_]
        Directional incidence matrix with shape (NE, N_nodes), where each row contains -1 for source,
        1 for target, and 0 elsewhere.
    - output_nodes_arr: NDArray[np.int_]
        Indices of output nodes where loss is applied.
    - loss: NDArray[np.float_]
        Array containing loss values for each output node.

    Returns:
    - grad_loss_vec: NDArray[np.float_]
        Gradient of the loss with respect to each edge pressure.
    """
    node_vec = np.concatenate((inputs_normalized, outputs_normalized))
    grad_loss_vec: NDArray[np.float_] = np.zeros([NE])
    first_output = np.size(inputs_normalized)
    # for idx in range(NE):
    #     output_idx = np.where(output_nodes_arr == np.where(DM[idx] == -1)[0][0])[0]  # index of output of edge
    #     x_j = node_vec[np.where(DM[idx] == 1)]
    #     y_i = node_vec[np.where(DM[idx] == -1)]
    #     loss_i = loss[0][output_idx[0]]
    #     grad_loss_ij = -(y_i-x_j)*loss_i
    #     grad_loss_vec[idx] = grad_loss_ij
    for idx in range(NE):
        # output_idx = np.where(output_nodes_arr == np.where(DM[idx] == -1)[0][0])[0]  # index of output of edge
        x_j = node_vec[np.where(DM[idx] == 1)]
        y_i = node_vec[np.where(DM[idx] == -1)]
        if np.where(DM[idx] == -1)[0][0] == first_output:
            loss_i = loss[0]
        elif np.where(DM[idx] == -1)[0][0] == first_output + 1:
            loss_i = loss[1]
        else:
            loss_i = 0
        grad_loss_ij = (y_i-x_j)*loss_i
        grad_loss_vec[idx] = grad_loss_ij
    return grad_loss_vec
