import torch
import pandas as pd

from qubit_sim.constants import DEVICE

from qubit_sim.noise_gen_classes import (
    OneOnFNoiseWithBumpNoiseGenerator,
    CombinedNoiseGenerator,
)

from .data_gen_sim_constants import (
    COLUNMS_LABELS,
    NUM_TIME_STEPS,
    TOTAL_PULSE_TRAINS,
    STANDARD_QUBIT_SIM,
    NUM_NOISE_REALIZATIONS,
    IDEAL_CPMG_CONTROL_PULSES,
)

torch.manual_seed(0)

noise_generator_130 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=130.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=140.0,
    flat_value=1 / 40,
)

noise_generator_150 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=150.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=150.0,
    flat_value=1 / 40,
)

noise_generator_170 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=170.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=180.0,
    flat_value=1 / 40,
)

noise_generator_190 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=190.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=200.0,
    flat_value=1 / 40,
)

noise_generator_210 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=210.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=220.0,
    flat_value=1 / 40,
)

noise_generator_230 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=230.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=240.0,
    flat_value=1 / 40,
)

one_thirty_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_130
)

one_fifty_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_150
)

one_seventy_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_170
)

one_ninety_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_190
)

two_ten_noise_generator = CombinedNoiseGenerator(x_noise_generator=noise_generator_210)

two_thirty_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_230
)

# divide by constant values so signal energy is all roughly the same
list_of_different_precomputed_noise = (4 / 3.25) * torch.concat(
    (
        one_thirty_noise_generator.precomputed_noise_combined_noise,
        one_fifty_noise_generator.precomputed_noise_combined_noise,
        one_seventy_noise_generator.precomputed_noise_combined_noise,
        one_ninety_noise_generator.precomputed_noise_combined_noise,
        two_ten_noise_generator.precomputed_noise_combined_noise,
        two_thirty_noise_generator.precomputed_noise_combined_noise,
    )
)

noise_profiles_names = [
    "130",
    "150",
    "170",
    "190",
    "210",
    "230",
]


def main() -> None:
    all_timesteps_control_unitaries = (
        STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
            control_pulse_time_series=IDEAL_CPMG_CONTROL_PULSES,
        )
    )

    alpha_beta_gamma_sols = STANDARD_QUBIT_SIM.compute_qfs_from_control_and_noise(
        all_timesteps_control_unitaries=all_timesteps_control_unitaries,
        noise=list_of_different_precomputed_noise.to(DEVICE),
    )

    data_frame = pd.DataFrame(
        alpha_beta_gamma_sols.cpu().numpy(),
        index=noise_profiles_names,
        columns=COLUNMS_LABELS,
    )

    data_frame.index.name = "Noise_Type"
    data_frame.to_csv("./data_csv_files/ideal_CPMG_data_r3.csv")


if __name__ == "__main__":
    main()
