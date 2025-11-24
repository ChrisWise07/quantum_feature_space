import torch
import numpy as np

from qubit_sim.constants import (
    LIST_OF_PAULI_EIGENSTATES,
    LIST_OF_PAULI_EIGENSTATES_TWO_QUBITS,
)

torch.manual_seed(0)
np.random.seed(0)


def return_conjugate_transpose_of_matrices(
    matrices: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the conjugate transpose of a batch of matrices.

    Args:
        matrices (Tensor): batch of matrices of shape (batch_size, 2, 2)

    Returns:
        Tensor: conjugate transpose of the batch of matrices
    """
    return matrices.conj().mT


def batch_based_matrix_trace(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the trace of a batch of matrices.

    Args:
        matrix (Tensor): batch of matrices of shape (batch_size, 2, 2)

    Returns:
        Tensor: trace of the batch of matrices
    """
    return matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def _return_qubit_initial_states_tensor(
    system_dimension: int,
) -> torch.Tensor:
    """
    A hidden function that returns the tensor of qubit initial states
    based on the system dimensions.

    Args:
        system_dimension: dimension of the system

    Returns:
        inital states: tensor of initial states of shape: (
            2^system_dimension - 1,
            system_dimension,
            system_dimension
        )
    """
    if system_dimension == 2:
        return LIST_OF_PAULI_EIGENSTATES

    return LIST_OF_PAULI_EIGENSTATES_TWO_QUBITS


def compute_ctrl_unitary_init_state_product(
    control_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the matrix product of the control unitaries with the
    initial states.

    Args:
        control_unitaries: tensor of shape
            (
                batch_size,
                system_dim,
                system_dim
            )

    Returns:
        Tensor of shape (batch_size, 6, system_dim, system_dim)
    """
    batch_size = control_unitaries.shape[0]

    initial_qubit_states = _return_qubit_initial_states_tensor(
        control_unitaries.shape[-1]
    )

    initial_rho_states = initial_qubit_states.repeat(batch_size, 1, 1, 1)
    control_unitaries_dagger = control_unitaries.conj().transpose(-1, -2).unsqueeze(1)
    control_unitaries = control_unitaries.unsqueeze(1)
    return control_unitaries @ initial_rho_states @ control_unitaries_dagger


def calculate_expectation_values(
    qfs_operators: torch.Tensor,
    control_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the expectation values of a set of Pauli operators.
    qfs_operators is a tensor of shape
    (num_operators, batch_size, system_dim, system_dim)
    representing different observables.

    Args:
        qfs_operators:
            tensor of shape:
                (
                    batch_size,
                    num_operators,
                    system_dimension,
                    system_dimension
                )
        control_unitaries:
            tensor of shape:
                (
                    batch_size,
                    system_dimension,
                    system_dimension
                )

    Returns:
        Tensor of shape (batch_size, num_operators, 6 ^ num_qubits).
    """
    control_based_evolution_matrices = compute_ctrl_unitary_init_state_product(
        control_unitaries
    )

    qdq_ctrl_unitary_product = control_based_evolution_matrices.unsqueeze(
        1
    ) @ qfs_operators.unsqueeze(2)

    return batch_based_matrix_trace(qdq_ctrl_unitary_product).real
