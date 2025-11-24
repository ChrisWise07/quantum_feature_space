import torch

from typing import Optional
from dataclasses import dataclass

from qubit_sim.qubit_sim_funcs import (
    construct_hamiltonian_for_each_timestep_noise_relisation_batchwise,
    return_exponentiated_scaled_hamiltonians,
    compute_unitary_at_timestep,
    create_interaction_hamiltonians_for_each_timestep_noise_relisation_batchwise,
    construct_batch_of_qfs_operators,
    compute_unitaries_for_all_time_steps_with_non_commuting_hamiltonians,
    compute_unitaries_for_all_time_steps_with_commuting_hamiltonians,
)


@dataclass
class QubitSimulator:
    """
    Class to simulate the evolution of a qubit under the influence of
    a control Hamiltonian and a noise Hamiltonian.

    Attributes:
        delta_t (float):
            The time step.
        control_static_operators (torch.Tensor):
            The static operators for the control Hamiltonian. Expected
            shape:
            (
                num_control_channels,
                system_dimension,
                system_dimension,
            )
        control_dynamic_operators (torch.Tensor):
            The dynamic operators for the control Hamiltonian. Expected
            shape:
            (
                num_control_channels,
                system_dimension,
                system_dimension,
            )
        noise_dynamic_operators (torch.Tensor):
            The dynamic operators for the noise Hamiltonian. Expected
            shape:
            (
                num_noise_channels,
                system_dimension,
                system_dimension,
            )
    """

    delta_t: float
    max_amp: float
    noise_dynamic_operators: torch.Tensor
    control_dynamic_operators: torch.Tensor
    control_static_operators: Optional[torch.Tensor] = None
    ideal_matrices: Optional[torch.Tensor] = None
    commuting_control_hamiltonians: bool = False
    commuting_noise_hamiltonians: bool = False

    def __post_init__(self):
        if self.commuting_control_hamiltonians:
            self.compute_unitaries_for_all_time_steps_func = (
                lambda h,
                dt: compute_unitaries_for_all_time_steps_with_commuting_hamiltonians(
                    h, dt
                )
            )

        else:
            self.compute_unitaries_for_all_time_steps_func = (
                lambda h,
                dt: compute_unitaries_for_all_time_steps_with_non_commuting_hamiltonians(
                    return_exponentiated_scaled_hamiltonians(h, dt)
                )
            )

    def set_max_amp(self, max_amp: float):
        """
        Sets the max amplitude of the control pulses.

        Args:
            max_amp (float):
                The max amplitude of the control pulses.
        """
        self.max_amp = max_amp

    def compute_all_timesteps_control_unitaries(
        self,
        control_pulse_time_series: torch.Tensor,
    ) -> torch.Tensor:
        total_num_examples = control_pulse_time_series.shape[0]

        batched_control_operators = self.control_dynamic_operators.repeat(
            total_num_examples, 1, 1, 1
        )

        if self.control_static_operators is not None:
            batched_static_operators = self.control_static_operators.repeat(
                total_num_examples, 1, 1, 1
            )
        else:
            batched_static_operators = None

        control_hamiltonians = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=control_pulse_time_series * self.max_amp,
                operators_for_time_evolving_elements=batched_control_operators,
                operators_for_static_elements=batched_static_operators,
            )
        )

        return self.compute_unitaries_for_all_time_steps_func(
            h=control_hamiltonians,
            dt=self.delta_t,
        )

    def compute_qfs_operators(
        self,
        all_timesteps_control_unitaries: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the qfs operators for the given control unitary
        time series.

        Args:
            all_timesteps_control_unitaries (torch.Tensor):
                The time series representing the pulses suggested by the
                model. Expected shape:
                (
                    total_num_examples,
                    num_time_steps,
                    system_dimension,
                    system_dimension,
                )
        Returns:
            qdq_dag_ops (torch.Tensor):
                The qdq_dag operators. Expected shape:
                (
                    total_num_examples,
                    num_time_steps,
                    system_dimension,
                    system_dimension,
                )
        """
        total_num_examples = all_timesteps_control_unitaries.shape[0]

        batched_noise_operators = self.noise_dynamic_operators.repeat(
            total_num_examples, 1, 1, 1
        )

        noise_hamiltonians = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=noise,
                operators_for_time_evolving_elements=batched_noise_operators,
            )
        )

        interaction_hamiltonians = create_interaction_hamiltonians_for_each_timestep_noise_relisation_batchwise(
            control_unitaries=all_timesteps_control_unitaries,
            noise_hamiltonians=noise_hamiltonians,
        )

        exponentiated_scaled_interaction_hamiltonians = (
            return_exponentiated_scaled_hamiltonians(
                hamiltonians=interaction_hamiltonians,
                delta_T=self.delta_t,
            )
        )

        final_timestep_interaction_unitaries = compute_unitary_at_timestep(
            exponential_hamiltonians=exponentiated_scaled_interaction_hamiltonians,
        )

        return construct_batch_of_qfs_operators(
            final_step_control_unitaries=all_timesteps_control_unitaries[:, -1],
            final_timestep_interaction_unitaries=final_timestep_interaction_unitaries,
        )

    def compute_qfs_from_control_and_noise(
        self,
        all_timesteps_control_unitaries: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simulate the evolution of a qubit under the influence of a
        control as described by the control pulse time series and a
        noise Hamiltonian. Returns the VO operators.

        Args:
            all_timesteps_control_unitaries (torch.Tensor):
                The time series representing the pulses suggested by the
                model. Expected shape:
                (
                    total_num_examples,
                    num_time_steps,
                    system_dimension,
                    system_dimension,
                )
        Returns:
            vo_operators (torch.Tensor):
                The VO operators. Expected shape:
                (
                    3,
                    total_num_examples,
                    num_time_steps,
                    system_dimension,
                    system_dimension,
                )
        """
        qfs_operators = self.compute_qfs_operators(
            all_timesteps_control_unitaries=all_timesteps_control_unitaries,
            noise=noise,
        )

        alpha_sols = torch.real(qfs_operators[:, :, 0, 1])
        beta_sols = -torch.imag(qfs_operators[:, :, 0, 1])
        gamma_sols = torch.real(qfs_operators[:, :, 0, 0])
        stacked = torch.stack((alpha_sols, beta_sols, gamma_sols), dim=2)
        return stacked.reshape(qfs_operators.shape[0], -1)
