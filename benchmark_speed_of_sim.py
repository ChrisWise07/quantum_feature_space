import time
import torch

from qubit_sim.qubit_sim_class import QubitSimulator
from qubit_sim.constants import SIGMA_X, SIGMA_Z, SIGMA_Y, DEVICE, QUBIT_ENERGY_GAP

from qubit_sim.utils import calculate_expectation_values

from qubit_sim.noise_gen_classes import (
    ColouredGaussianNoiseGenerator,
    CombinedNoiseGenerator,
)

from data_gen.data_gen_sim_constants import (
    NUM_TIME_STEPS,
    STANDARD_QUBIT_SIM,
    TOTAL_TIME,
    MAX_AMP,
    NOISE_DYNAMIC_OPERATORS,
)

torch.manual_seed(0)

number_of_noise_realizations = 2000
total_examples = 500

coloured_noise_ns = ColouredGaussianNoiseGenerator(
    number_of_noise_realisations=number_of_noise_realizations,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=total_examples,
    device=DEVICE,
    precompute_noise=False,
)

coloured_noise_ns_combined_noise = CombinedNoiseGenerator(
    x_noise_generator=coloured_noise_ns,
)

expectation_values = torch.empty((total_examples, 3, 6))


STANDARD_QUBIT_SIM = QubitSimulator(
    max_amp=MAX_AMP,
    delta_t=TOTAL_TIME / NUM_TIME_STEPS,
    noise_dynamic_operators=NOISE_DYNAMIC_OPERATORS,
    control_static_operators=0.5 * QUBIT_ENERGY_GAP * SIGMA_Z,
    control_dynamic_operators=torch.stack((0.5 * SIGMA_X, 0.5 * SIGMA_Y), dim=0),
)


def main() -> None:
    batch_size = 7
    num_batches = total_examples // batch_size + 1
    start = time.time()

    for i in range(num_batches):
        print(f"Batch {i + 1}/{num_batches}")
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_examples)
        total_size = end_index - start_index

        control_pulse_sequences = (
            2
            * torch.rand(
                (total_size, 1024, 2),
            )
            - 1
        )

        all_timesteps_control_unitaries = (
            STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
                control_pulse_sequences.to(DEVICE),
            )
        )

        noise = coloured_noise_ns_combined_noise.generate_new_noise_instance(
            total_num_examples=total_size,
        )

        qdq_dagger_matrices = STANDARD_QUBIT_SIM.compute_qdq_dag_operators(
            all_timesteps_control_unitaries, noise
        )

        expectation_values[start_index:end_index] = calculate_expectation_values(
            qdq_dagger_matrices=qdq_dagger_matrices,
            control_unitaries=all_timesteps_control_unitaries[:, -1],
        )

    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
