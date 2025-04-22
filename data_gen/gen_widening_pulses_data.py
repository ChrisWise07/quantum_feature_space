import torch
import pandas as pd

from qubit_sim.constants import DEVICE
from qubit_sim.control_pulse_funcs import generate_gaussian_pulses

from qubit_sim.noise_gen_classes import (
    OneOnFNoiseWithBumpNoiseGenerator,
    CombinedNoiseGenerator,
)

from .data_gen_sim_constants import (
    TOTAL_TIME,
    COLUNMS_LABELS,
    NUM_TIME_STEPS,
    TIME_STEP_VALUES,
    TOTAL_PULSE_TRAINS,
    STANDARD_QUBIT_SIM,
    NUM_PULSES_PER_TRAIN,
    NUM_NOISE_REALIZATIONS,
    IDEAL_CPMG_CONTROL_PULSES,
    NUM_CONTROL_CHANNELS,
)

torch.manual_seed(0)

gaussian_scale_widths = torch.tensor([1 / 3, 1 / 6, 1 / 12, 1 / 24, 1 / 48])

gaussian_centre_positions = (
    (torch.arange(1, NUM_PULSES_PER_TRAIN + 1) - 0.5)
    / NUM_PULSES_PER_TRAIN
    * TOTAL_TIME
)[None, :, None].repeat(len(gaussian_scale_widths), 1, NUM_CONTROL_CHANNELS)

gaussian_amplitudes = torch.full(
    (len(gaussian_scale_widths), NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS), torch.pi
)

std_values = 0.5 * TOTAL_TIME / NUM_PULSES_PER_TRAIN * gaussian_scale_widths

gaussian_std_values = std_values.view(-1, 1, 1).expand(
    -1, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS
)

gaussian_pulse_parameters = torch.stack(
    (gaussian_amplitudes, gaussian_centre_positions, gaussian_std_values), dim=-1
).view(len(gaussian_scale_widths), NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS * 3)

guassian_control_pulses = generate_gaussian_pulses(
    number_of_channels=NUM_CONTROL_CHANNELS,
    time_range_values=TIME_STEP_VALUES,
    pulse_parameters=gaussian_pulse_parameters.to(DEVICE),
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

total_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator,
)

precomputed_noise_scaled = 4 * (
    total_noise_generator.precomputed_noise_combined_noise / 3.25
).to(DEVICE)


def main() -> None:
    all_timesteps_control_unitaries = (
        STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
            control_pulse_time_series=guassian_control_pulses,
        )
    )

    alpha_beta_gamma_sols = (
        STANDARD_QUBIT_SIM.compute_alpha_beta_gamma_sols_for_control_and_noise(
            all_timesteps_control_unitaries=all_timesteps_control_unitaries,
            noise=precomputed_noise_scaled.to(DEVICE),
        )
    )

    data_frame = pd.DataFrame(
        alpha_beta_gamma_sols.cpu().numpy(),
        index=gaussian_scale_widths.numpy(),
        columns=COLUNMS_LABELS,
    )

    data_frame.index.name = "Pulse Widths"
    data_frame.to_csv("./data_csv_files/data_for_widening_pulses.csv")


if __name__ == "__main__":
    main()
