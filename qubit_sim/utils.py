import torch
import numpy as np

from typing import Tuple

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


def compute_fidelity(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Calculates the fidelity between two unitary matrices.

    Args:
        U (torch.Tensor): A unitary matrix.
        V (torch.Tensor): Another unitary matrix.

    Returns:
        float: The fidelity between U and V.
    """
    U_dagger = U.conj().transpose(-1, -2)

    trace = batch_based_matrix_trace(torch.matmul(U_dagger, V))

    fidelity = torch.abs(trace) ** 2

    return fidelity


def compute_nuc_norm_of_diff_between_batch_of_matrices(
    rho: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the nuclear norm of the difference between two batches of
    density matrices.

    Args:
        rho (np.ndarray): density matrix
        sigma (np.ndarray): density matrix

    Returns:
        Tensor: trace distance
    """
    return torch.linalg.matrix_norm(rho - sigma, ord="nuc")


def fidelity_batch_of_matrices(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Calculates the fidelity between two batches of density matrices.

    Args:
        rho (torch.Tensor): density matrix
        sigma (torch.Tensor): density matrix

    Returns:
        Tensor: fidelity
    """
    rho_dagger = rho.conj().transpose(-1, -2)
    sigma_dagger = sigma.conj().transpose(-1, -2)

    return torch.square(
        torch.abs(
            batch_based_matrix_trace(torch.matmul(rho_dagger, sigma))
            / (
                torch.sqrt(
                    batch_based_matrix_trace(torch.matmul(rho_dagger, rho))
                    * batch_based_matrix_trace(torch.matmul(sigma_dagger, sigma))
                )
                + 1e-6
            )
        )
    )


def calculate_psi_theta_mu(
    qdq_dagger: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Calculate the parameters psi, theta, mu from the QDQ^{\dagger}
    matrix.

    Args:
        matrix (Tensor):
            QDQ^{\dagger} matrix of shape (batch_size, 2, 2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: psi, theta, mu
    """

    mu = torch.abs(torch.real(torch.linalg.eigvals(qdq_dagger)))[:, :, 0]

    theta = 0.5 * torch.acos(torch.real(qdq_dagger[:, :, 0, 0]) / mu)
    theta = theta.nan_to_num(0)

    psi = 0.5 * torch.imag(
        torch.log(qdq_dagger[:, :, 0, 1] / (-mu * torch.sin(2 * theta)))
    )

    psi = psi.nan_to_num(0)
    return psi, theta, mu


def calculate_qdq_dag_parameters(
    qdq_dagger_operators: torch.Tensor,
) -> torch.Tensor:
    r"""
    Calculate the parameters psi, theta, mu from the QDQ^{\dagger}
    operators.

    Args:
        qdq_dagger_operators (Tensor):
            QDQ^{\dagger} operators of shape (batch_size, 3, 2, 2)

    Returns:
        Tensor: ground truth parameters in the format:
        [
            x_psi,
            x_theta,
            x_mu,
            y_psi,
            y_theta,
            y_mu,
            z_psi,
            z_theta,
            z_mu,
        ]
    """
    psi, theta, mu = calculate_psi_theta_mu(qdq_dagger_operators)
    return torch.stack([psi, theta, mu], dim=-1).reshape(-1, 9)


def calculate_trig_expo_funcs_for_batch(
    psi: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    r"""
    Construct the estimated noise encoding matrix V_O from the
    parameters psi, theta, mu and the inverse of the pauli observable O.

    Args:
        psi (Tensor): parameter of shape (batch_size,)
        theta (Tensor): parameter of shape (batch_size,)

    Returns:
        Tensor:
            estimated noise encoding matrix QDQ^{\dagger} of shape (
                batch_size, 2, 2
            )
    """
    cos_2theta = torch.cos(2 * theta)
    sin_2theta = torch.sin(2 * theta)
    exp_2ipsi = torch.exp(2j * psi)
    exp_minus2ipsi = torch.exp(-2j * psi)

    return torch.stack(
        [
            cos_2theta,
            -exp_2ipsi * sin_2theta,
            -exp_minus2ipsi * sin_2theta,
            -cos_2theta,
        ],
        dim=-1,
    ).reshape(-1, 2, 2)


def calculate_QDQdagger(
    batch_parameters: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Construct QDQ^{\dagger} for a batch of parameters. The function
    assumes the following order
    (x_psi, x_theta, x_mu, y_psi, y_theta, y_mu, z_psi, z_theta, z_mu).
    Return the matrices for the x, y and z axis, in that order.

    Args:
        batch_parameters (Tensor): batch of parameters

    Returns:
        Tensor: calculated QDQ^{\dagger} matrices in the order x, y, z
    """
    (
        x_psi,
        x_theta,
        x_mu,
        y_psi,
        y_theta,
        y_mu,
        z_psi,
        z_theta,
        z_mu,
    ) = batch_parameters.unbind(dim=1)

    return (
        calculate_trig_expo_funcs_for_batch(x_psi, x_theta) * x_mu[:, None, None],
        calculate_trig_expo_funcs_for_batch(y_psi, y_theta) * y_mu[:, None, None],
        calculate_trig_expo_funcs_for_batch(z_psi, z_theta) * z_mu[:, None, None],
    )


def return_estimated_VO_unital_for_batch(
    batch_parameters: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct the estimated noise encoding matrices V_O for a batch of
    parameters.

    Args:
        batch_parameters: batch of parameters

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            estimated V_O matrices for the x, y and z axis, in that
            order
    """
    (
        x_QDQdagger,
        y_QDQdagger,
        z_QDQdagger,
    ) = calculate_QDQdagger(batch_parameters)

    return (
        SIGMA_X @ x_QDQdagger,
        SIGMA_Y @ y_QDQdagger,
        SIGMA_Z @ z_QDQdagger,
    )


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


def create_control_coefficients_matrix(
    ctrl_unitary_init_state_product: torch.Tensor,
) -> torch.Tensor:
    """
    Create the control coefficients matrix used to find the qdq_dagger
    parameters.

    Args:
        ctrl_unitary_init_state_product: tensor of shape (
            batch_size, 6, system_dim, system_dim
        )

    Returns:
        Tuple of tensors of shape (batch_size, 6, 3)
    """
    return torch.stack(
        [
            2 * ctrl_unitary_init_state_product[:, :, 0, 1].real,
            2 * -ctrl_unitary_init_state_product[:, :, 0, 1].imag,
            2 * ctrl_unitary_init_state_product[:, :, 0, 0].real - 1,
        ],
        dim=-1,
    )


def calculate_expectation_values(
    qdq_dagger_matrices: torch.Tensor,
    control_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the expectation values of a set of Pauli operators.
    qdq_dagger_matrices is a tensor of shape
    (num_operators, batch_size, system_dim, system_dim)
    representing different observables.

    Args:
        qdq_dagger_matrices:
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

    qdq_ctrl_unitary_product = qdq_dagger_matrices.unsqueeze(
        2
    ) @ control_based_evolution_matrices.unsqueeze(1)

    return batch_based_matrix_trace(qdq_ctrl_unitary_product).real


def calculate_qdq_dagger_parameters(
    ctrl_coefficients_matrix: torch.Tensor,
    expectations: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the parameters alpha, beta, gamma for the qdq_daggers
    matrices one for each axis.

    Args:
        ctrl_coefficients_matrix: tensor of shape:
            (batch_size, 6, 3)
        expectation_values: tensor of shape:
            (batch_size, num_operators, 6 ^ num_qubits)

    Returns:
        Tensor of shape (batch_size, num_operators, 3)
    """
    ctrl_coefficients_matrix_inverse = torch.linalg.pinv(ctrl_coefficients_matrix)

    return (
        ctrl_coefficients_matrix_inverse.unsqueeze(1) @ expectations.unsqueeze(-1)
    ).squeeze(-1)


def construct_qdq_operators_from_parameters(
    qdq_dagger_coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Use the qdq_dagger_coefficients to construct the qdq_dagger
    operators.

    Args:
        qdq_dagger_coefficients: tensor of shape
            (batch_size, num_operators, 3)

    Returns:
        Tensor of shape (batch_size, num_operators, 2, 2)
    """

    alpha, beta, gamma = qdq_dagger_coefficients.unbind(dim=-1)

    batch_size = qdq_dagger_coefficients.shape[0]

    num_operators = qdq_dagger_coefficients.shape[1]

    return torch.stack(
        [
            gamma,
            alpha + 1j * beta,
            alpha - 1j * beta,
            -gamma,
        ],
        dim=-1,
    ).view(batch_size, num_operators, 2, 2)


def calculate_quantum_features_from_expectation_values(
    expectation_values: torch.Tensor, control_unitaries: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates Vx, Vy, Vz from expectation values and control unitaries.

    Args:
        expectation_values:
            expectation values of the shape (batch_size, 3, 2, 2)
        control_unitaries: control unitaries

    Returns:
        List of Vx, Vy, Vz

    """
    ctrl_unitary_init_state_product = compute_ctrl_unitary_init_state_product(
        control_unitaries
    )

    ctrl_coefficients_matrix = create_control_coefficients_matrix(
        ctrl_unitary_init_state_product
    )

    return calculate_qdq_dagger_parameters(ctrl_coefficients_matrix, expectation_values)


def calculate_state_from_observable_expectations(
    expectation_values: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the state, that is, peform state tomography, from the
    expectation values and observables.

    Args:
        expectation_values of shape: (
            batch_size, num_observables, 6 * num_qubits
        )

    Returns:
        Tensor: state of the system with shape:
            (
                batch_size,
                system_dimension,
                system_dimension
            )
    """
    if expectation_values.shape[1] == 3:
        observables = COMBINED_SIGMA_TENSOR
        identity_expanded = SIGMA_I

    if expectation_values.shape[1] == 15:
        observables = COMBINED_SIGMA_TENSOR_TWO_QUBITS
        identity_expanded = torch.kron(SIGMA_I, SIGMA_I)

    observables_expanded = observables.unsqueeze(2)
    identity_expanded = identity_expanded.unsqueeze(0)
    expectation_values_expanded = expectation_values.unsqueeze(-1).unsqueeze(-1)

    return (
        1
        / 2
        * (
            identity_expanded
            + torch.sum(
                observables_expanded * expectation_values_expanded,
                dim=1,
            )
        )
    )


def compute_process_matrix_for_single_qubit(
    rho_zero: torch.Tensor,
    rho_one: torch.Tensor,
    rho_plus: torch.Tensor,
    rho_minus_i: torch.Tensor,
) -> torch.Tensor:
    """
    Following pg. 393 Box 8.5 of Nielsen and Chuang, calculate the chi
    matrix, i.e. the process matrix, for a single qubit. Note in
    Nielsen and Chuang, the state |-><-| is used. However this is a
    mistake and should be |i-><i|.

    Args:
        rho_zero:
            the density matrix of the |0><0| after the quantum process.
            expected shape: (batch_size, 2, 2)
        rho_one:
            the density matrix of the |1><1| after the quantum process.
            expected shape: (batch_size, 2, 2)
        rho_plus:
            the density matrix of the |+><+| after the quantum process.
            expected shape: (batch_size, 2, 2)
        rho_minus_i:
            the density matrix of the |i-><i-| after the quantum process.
            expected shape: (batch_size, 2, 2)

    Returns:
        Tensor: chi/process matrix of shape (batch_size, 4, 4)
    """
    batch_size = rho_zero.shape[0]
    rho_one_prime = rho_zero
    rho_four_prime = rho_one

    rho_two_prime = (
        rho_plus - 1j * rho_minus_i - (1 - 1j) * (rho_one_prime + rho_four_prime) / 2
    )

    rho_three_prime = (
        rho_plus + 1j * rho_minus_i - (1 + 1j) * (rho_one_prime + rho_four_prime) / 2
    )

    rho_for_chi = torch.zeros((batch_size, 4, 4), dtype=torch.cfloat, device=DEVICE)

    rho_for_chi[:, 0:2, 0:2] = rho_one_prime
    rho_for_chi[:, 0:2, 2:4] = rho_two_prime
    rho_for_chi[:, 2:4, 0:2] = rho_three_prime
    rho_for_chi[:, 2:4, 2:4] = rho_four_prime

    return LAMBDA_MATRIX_FOR_CHI_MATRIX @ rho_for_chi @ LAMBDA_MATRIX_FOR_CHI_MATRIX


def compute_process_matrix_from_expectations(
    expectations: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the process matrix from the expectations of the Pauli
    operators.

    Args:
        expectations:
            expectation values of the form
            [
                x_rho_1,
                y_rho_1,
                z_rho_1,
                ...,
                x_rho_n,
                y_rho_n,
                z_rho_n
            ]
            where n is the number of expectations. Expected shape: (
                batch_size, n
            )

    Returns:
        Tensor: process matrix with shape (batch_size, 4, 4)
    """
    reconstructed_states = calculate_state_from_observable_expectations(expectations)

    reconstructed_ones = reconstructed_states[:, 5]
    reconstructed_zeros = reconstructed_states[:, 4]
    reconstructed_plus = reconstructed_states[:, 0]
    reconstructed_minus_i = reconstructed_states[:, 3]

    return compute_process_matrix_for_single_qubit(
        rho_zero=reconstructed_zeros,
        rho_one=reconstructed_ones,
        rho_plus=reconstructed_plus,
        rho_minus_i=reconstructed_minus_i,
    )


def compute_process_fidelity(
    process_matrix_one: torch.Tensor,
    process_matrix_two: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the process fidelity between two process matrices. Note
    this method assumes one of the process matrices comes from a unitary
    process.

    Args:
        process_matrix_one:
            process matrix one with shape:
                (
                    batch_size,
                    system_dim ^ 2,
                    system_dim ^ 2
                )
        process_matrix_two:
            process matrix two with shape:
                (
                    batch_size,
                    system_dim ^ 2,
                    system_dim ^ 2
                )

    Returns:
        Tensor: process fidelity with shape (batch_size,)
    """
    return batch_based_matrix_trace(
        torch.matmul(process_matrix_one, process_matrix_two)
    ).real
