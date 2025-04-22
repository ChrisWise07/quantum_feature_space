import torch
import pandas as pd

from qubit_sim.constants import PI, DEVICE
from qubit_sim.control_pulse_funcs import generate_gaussian_pulses

from qubit_sim.noise_gen_classes import (
    OneOnFNoiseWithBumpNoiseGenerator,
    CombinedNoiseGenerator,
)

from qubit_sim.qubit_sim_class import QubitSimulator

from .data_gen_sim_constants import (
    TIME_STEP,
    TOTAL_TIME,
    COLUNMS_LABELS,
    NUM_TIME_STEPS,
    TOTAL_PULSE_TRAINS,
    STANDARD_QUBIT_SIM,
    NUM_CONTROL_CHANNELS,
    NUM_PULSES_PER_TRAIN,
    NUM_NOISE_REALIZATIONS,
)


def generate_randomised_gaussian_centre_positions(
    num_diff_control_pulses: int = 1,
    total_time: float = TOTAL_TIME,
    num_time_steps: int = NUM_TIME_STEPS,
    num_pulses_per_train: int = NUM_PULSES_PER_TRAIN,
    num_control_channels: int = NUM_CONTROL_CHANNELS,
):
    shift_limit = 24 * total_time / num_time_steps
    n = torch.arange(1, num_pulses_per_train + 1)
    tau_n = (n - 0.5) / num_pulses_per_train * total_time
    tau_n_expanded = tau_n[None, :, None]

    random_shifts = shift_limit * (
        2
        * torch.rand(
            (num_diff_control_pulses, num_pulses_per_train, num_control_channels)
        )
        - 1
    )

    return tau_n_expanded + random_shifts


torch.manual_seed(0)
num_diff_control_pulses = 50
scale = 1 / 24
std_value = TOTAL_TIME / NUM_PULSES_PER_TRAIN * scale

gaussian_centre_positions = generate_randomised_gaussian_centre_positions(
    total_time=TOTAL_TIME,
    num_time_steps=NUM_TIME_STEPS,
    num_pulses_per_train=NUM_PULSES_PER_TRAIN,
    num_diff_control_pulses=num_diff_control_pulses,
    num_control_channels=NUM_CONTROL_CHANNELS,
)

gaussian_amplitudes = torch.full(
    (num_diff_control_pulses, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS), PI
)

small_amplitudes_fluctuations = (
    0.20
    * torch.pi
    * (
        2
        * torch.rand(
            (num_diff_control_pulses, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS)
        )
        - 1
    )
)

gaussian_amplitudes = gaussian_amplitudes + small_amplitudes_fluctuations

gaussian_std_values = torch.full(
    (num_diff_control_pulses, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS), std_value
)

gaussian_pulse_parameters = torch.stack(
    (gaussian_amplitudes, gaussian_centre_positions, gaussian_std_values), dim=-1
)

pulse_parameters = gaussian_pulse_parameters.reshape(
    num_diff_control_pulses, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS * 3
)

time_step_values = torch.linspace(0, TOTAL_TIME - TIME_STEP, NUM_TIME_STEPS).to(DEVICE)

control_pulse = generate_gaussian_pulses(
    number_of_channels=NUM_CONTROL_CHANNELS,
    time_range_values=time_step_values,
    pulse_parameters=pulse_parameters.to(DEVICE),
)

noise_generator = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=200.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=210.0,
    flat_value=1 / 40,
)

total_noise_generator = CombinedNoiseGenerator(x_noise_generator=noise_generator)

precomputed_noise_scaled = 4 * (
    total_noise_generator.precomputed_noise_combined_noise / 3.25
).to(DEVICE)


def main() -> None:
    batch_size = 7
    num_batches = num_diff_control_pulses // batch_size + 1

    all_timesteps_control_unitaries = (
        STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
            control_pulse_time_series=control_pulse,
        )
    )

    all_timesteps_control_unitaries = all_timesteps_control_unitaries.to("cpu")
    alpha_beta_gamma_sols = torch.empty(all_timesteps_control_unitaries.shape[0], 9)

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size

        alpha_beta_gamma_sols[start:end] = (
            STANDARD_QUBIT_SIM.compute_alpha_beta_gamma_sols_for_control_and_noise(
                all_timesteps_control_unitaries=all_timesteps_control_unitaries[
                    start:end
                ].to(DEVICE),
                noise=precomputed_noise_scaled,
            )
        )

    data_frame = pd.DataFrame(
        alpha_beta_gamma_sols.cpu().numpy(),
        columns=COLUNMS_LABELS,
    )

    data_frame.index.name = "Serial Number"
    data_frame.to_csv("./data_csv_files/real_CPMG_data.csv")


if __name__ == "__main__":
    main()
