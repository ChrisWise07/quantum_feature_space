import torch

from typing import Optional
from .utils import return_conjugate_transpose_of_matrices

from .constants import (
    COMBINED_SIGMA_TENSOR_TWO_QUBITS,
    COMBINED_SIGMA_TENSOR,
)

NOISE_DIM = 2


def construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
    time_evolving_elements: torch.Tensor,
    operators_for_time_evolving_elements: torch.Tensor,
    operators_for_static_elements: Optional[torch.Tensor] = None,
):
    """
    Constructs a tensor of Hamiltonians for each batch, and timestep
    witin the batch, and each noise realisation of that batch and
    timestep. Multiples in an efficient manner, see the code for
    details.

    Args:
        time_evolving_elements (torch.Tensor):
            the time evolving elements such as control fields or noise
            processes. Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                number_of_dynamic_operators
            )
        operators_for_time_evolving_elements (torch.Tensor):
            the operators that are multiplied with the time evolving
            elements. Expected Shape:
            (
                batch_size,
                number_of_dynamic_operators,
                system_dimension,
                system_dimension
            )
        operators_for_static_elements (torch.Tensor):
            the operators that are multiplied with the static elements
            that are not time evolving such as the energy gap.
            Expected Shape:
            (
                batch_size,
                number_of_static_operators,
                system_dimension,
                system_dimension
            )
        operators_dim (int, optional):
            The dimensions that the operators are in. Defaults to -3.


    Returns:
        torch.Tensor:
            Resulting Hamiltonians of shape:
            (
                batch_size,
                num_timesteps,
                number_of_noise_realisations (optional),
                system_dimension,
                system_dimension
            )
    """
    dim_time_evolving_elements = time_evolving_elements.dim()
    expand_amount = dim_time_evolving_elements - 2
    operators_dim = dim_time_evolving_elements - 1
    time_evolving_elements_expanded = time_evolving_elements[..., None, None]

    operators_for_time_evolving_elements_new_shape = (
        operators_for_time_evolving_elements.shape[0:1]
        + (1,) * expand_amount
        + operators_for_time_evolving_elements.shape[1:]
    )

    operators_for_time_evolving_elements_expanded = (
        operators_for_time_evolving_elements.view(
            operators_for_time_evolving_elements_new_shape
        )
    )

    dynamic_part = torch.sum(
        time_evolving_elements_expanded * operators_for_time_evolving_elements_expanded,
        dim=operators_dim,
    )

    if operators_for_static_elements is None:
        return dynamic_part

    operators_for_static_elements_new_shape = (
        operators_for_static_elements.shape[0:1]
        + (1,) * expand_amount
        + operators_for_static_elements.shape[1:]
    )

    operators_for_static_elements_expanded = operators_for_static_elements.view(
        operators_for_static_elements_new_shape
    )

    static_part = torch.sum(operators_for_static_elements_expanded, dim=operators_dim)

    return dynamic_part + static_part


def return_exponentiated_scaled_hamiltonians(
    hamiltonians: torch.Tensor,
    delta_T: float,
) -> torch.Tensor:
    """
    Computes the exponential of the scaled Hamiltonian for the given
    batch of Hamiltonians. Hamiltonians are scaled by the time step
    delta_T. This is to be used for the unitary evolution operator
    Trotter-Suzuki decomposition.

    Args:
        hamiltonians (torch.Tensor):
            Hamiltonian for which to compute the evolution operator.
            Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
        delta_T (float):
            Time step for the evolution.

    Returns:
        torch.Tensor:
            Resulting exponential of the scaled Hamiltonian of shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
    """
    scaled_hamiltonian = hamiltonians * (-1j * delta_T)
    return torch.linalg.matrix_exp(scaled_hamiltonian)


def compute_unitary_at_timestep(
    exponential_hamiltonians: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the unitary at a given timestep for the given batch of
    exponential Hamiltonians. Computes for the whole batch and for each
    noise realisation. This uses the Trotter-Suzuki decomposition. Note
    you should only pass in the exponential Hamiltonians actually
    needed for the time step, i.e. if you want to compute the unitary
    at time step t, then you should only pass in the exponential
    Hamiltonians for time steps 0 to t.

    Args:
        exponential_hamiltonians (torch.Tensor):
            Exponential Hamiltonian for which to compute the evolution
            operator. Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting unitary evolution operators of shape:
            (
                batch_size,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
    """
    (
        _,
        num_time_steps,
        *_,
    ) = exponential_hamiltonians.shape

    if num_time_steps == 1:
        return exponential_hamiltonians[:, 0]

    if num_time_steps % 2 == 1:
        last_matrix = exponential_hamiltonians[:, -1:]
        exponential_hamiltonians = exponential_hamiltonians[:, :-1]
    else:
        last_matrix = None

    even_exponential_hamiltonians = exponential_hamiltonians[:, 0::2]
    odd_exponential_hamiltonians = exponential_hamiltonians[:, 1::2]

    product = torch.matmul(odd_exponential_hamiltonians, even_exponential_hamiltonians)

    if last_matrix is not None:
        product = torch.cat([product, last_matrix], dim=1)

    return compute_unitary_at_timestep(product)


def compute_unitaries_for_all_time_steps(
    exponential_hamiltonians: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the unitaries for all time steps for the given batch of
    exponentiated Hamiltonians. Computes for the whole batch and for
    each noise realisation. This uses the Trotter-Suzuki decomposition.

    Use of clone is necessary to enable gradient computation.

    Args:
        exponential_hamiltonians (torch.Tensor):
            Exponential Hamiltonian for which to compute the evolution
            operator. Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting unitary evolution operators of shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
    """

    result_unitaries = torch.empty_like(exponential_hamiltonians)

    result_unitaries[:, 0, ...] = exponential_hamiltonians[:, 0, ...]

    for time_step in range(1, exponential_hamiltonians.shape[1]):
        result_unitaries[:, time_step, ...] = (
            exponential_hamiltonians[:, time_step, ...]
            @ result_unitaries[:, time_step - 1, ...].clone()
        )

    return result_unitaries


def create_interaction_hamiltonian_for_each_timestep_noise_relisation_batchwise(
    control_unitaries: torch.Tensor,
    noise_hamiltonians: torch.Tensor,
) -> torch.Tensor:
    r"""
    Creates the interaction Hamiltonian for the given batch of control
    unitaries and noise Hamiltonians. Performs the calculation:
    H = U_ctrl^\dagger * H_noise * U_ctrl
    where U_dagger is the conjugate transpose of the control
    unitary and U_ctrl is the control unitary.

    Args:
        control_unitaries (torch.Tensor):
            Unitary evolution operator for the control part of the
            Hamiltonian. Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                system_dimension,
                system_dimension
            )
        noise_hamiltonians (torch.Tensor):
            Hamiltonian for the noise part of the Hamiltonian.
            Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_realisations,
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting interaction Hamiltonian of shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_realisations,
                system_dimension,
                system_dimension
            )
    """
    control_unitaries_expanded = control_unitaries.unsqueeze(2)

    control_unitaries_expanded_dagger = torch.conj(
        control_unitaries_expanded
    ).transpose(-1, -2)

    return torch.matmul(
        control_unitaries_expanded_dagger,
        torch.matmul(noise_hamiltonians, control_unitaries_expanded),
    )


def __return_observables_for_vo_construction(
    system_dimension: int,
) -> torch.Tensor:
    """
    Returns the observables for the construction of the VO operator.
    Handles reshaping of the observables and if the signle or two qubit
    case is being considered.

    Args:
        system_dimension (int):
            The dimension of the system, i.e. the qubit

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            The observables for the construction of the VO operator.
            The first tensor is the observables constructing the
            ensemble average and so has an extra dimension for the
            number of realisations. The second tensor is the
            observables for the left multiplication of the ensemble
            average and so does not have that extra dimension. The shape
            of the first tensor is:
            (
                number_of_observables,
                1,
                1,
                system_dimension,
                system_dimension
            )
            The shape of the second tensor is:
            (
                number_of_observables,
                1,
                system_dimension,
                system_dimension
            )
    """
    if system_dimension == 2:
        return COMBINED_SIGMA_TENSOR

    return COMBINED_SIGMA_TENSOR_TWO_QUBITS


def construct_batch_of_q_d_q_dag_operators(
    final_step_control_unitaries: torch.Tensor,
    final_step_interaction_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Constructs the QDQ_dag operators for the given batch of unitary
    interaction operators. Note the vo operator are stored in the
    order of X operator, Y operator, Z operator.

    Args:
        final_step_control_unitaries (torch.Tensor):
            Unitary evolution operator for the control part of the
            Hamiltonian. Expected Shape:
            (
                batch_size,
                system_dimension,
                system_dimension
            )
        final_step_interaction_unitaries (torch.Tensor):
            Unitary evolution operator for the interaction part of the
            Hamiltonian. Expected Shape:
            (
                batch_size,
                number_of_realisations,
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting QDQ_dag operators of shape:
            (
                batch_size,
                3
                system_dimension,
                system_dimension
            )
    """
    final_step_control_unitaries_expanded = final_step_control_unitaries.unsqueeze(1)

    final_step_control_unitaries_expanded_dagger = (
        return_conjugate_transpose_of_matrices(final_step_control_unitaries_expanded)
    )

    final_step_interaction_unitaries_tilde = (
        final_step_control_unitaries_expanded
        @ final_step_interaction_unitaries
        @ final_step_control_unitaries_expanded_dagger
    )

    final_step_interaction_unitaries_tilde_expanded = (
        final_step_interaction_unitaries_tilde.unsqueeze(1)
    )

    final_step_interaction_unitaries_tilde_dagger = (
        return_conjugate_transpose_of_matrices(
            final_step_interaction_unitaries_tilde_expanded
        )
    )

    observables = __return_observables_for_vo_construction(
        final_step_control_unitaries.shape[-1]
    ).unsqueeze(2)

    return (
        final_step_interaction_unitaries_tilde_dagger
        @ observables
        @ final_step_interaction_unitaries_tilde_expanded
    ).mean(dim=NOISE_DIM)


def return_alpha_beta_sols_from_qdq_dag_operators(
    qdq_dag_operators: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the alpha, beta, gamma solutions from the given batch of
    QDQ_dag operators. The QDQ_dag operators are assumed to be in the
    order of X operator, Y operator, Z operator.

    Args:
        qdq_dag_operators (torch.Tensor):
            QDQ_dag operators. Expected Shape:
            (
                batch_size,
                3,
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            The alpha, beta, gamma solutions. Expected Shape:
            (
                batch_size,
                3,
                2,
                2
            )
    """
    return qdq_dag_operators.reshape(qdq_dag_operators.shape[0], 3, 2, 2)
