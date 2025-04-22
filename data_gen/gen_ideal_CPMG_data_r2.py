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

noise_generator_15 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=15.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=25.0,
    flat_value=1 / 40,
)

noise_generator_30 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=30.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=40.0,
    flat_value=1 / 40,
)

noise_generator_60 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=60.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=70.0,
    flat_value=1 / 40,
)

noise_generator_120 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=120.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=130.0,
    flat_value=1 / 40,
)

noise_generator_240 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=240.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=250.0,
    flat_value=1 / 40,
)

noise_generator_480 = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=480.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=480.0,
    flat_value=1 / 40,
)

fifteeen_noise_generator = CombinedNoiseGenerator(x_noise_generator=noise_generator_15)
thirty_noise_generator = CombinedNoiseGenerator(x_noise_generator=noise_generator_30)
sixty_noise_generator = CombinedNoiseGenerator(x_noise_generator=noise_generator_60)

one_twenty_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_120
)

two_forty_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_240
)

four_eighty_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=noise_generator_480
)

# divide by constant values so signal energy is all roughly the same
list_of_different_precomputed_noise = (4 / 3.25) * torch.concat(
    (
        fifteeen_noise_generator.precomputed_noise_combined_noise,
        thirty_noise_generator.precomputed_noise_combined_noise,
        sixty_noise_generator.precomputed_noise_combined_noise,
        one_twenty_noise_generator.precomputed_noise_combined_noise,
        two_forty_noise_generator.precomputed_noise_combined_noise,
        four_eighty_noise_generator.precomputed_noise_combined_noise,
    )
)

noise_profiles_names = [
    "15",
    "30",
    "60",
    "120",
    "240",
    "480",
]


def main() -> None:
    all_timesteps_control_unitaries = (
        STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
            control_pulse_time_series=IDEAL_CPMG_CONTROL_PULSES,
        )
    )

    alpha_beta_gamma_sols = (
        STANDARD_QUBIT_SIM.compute_alpha_beta_gamma_sols_for_control_and_noise(
            all_timesteps_control_unitaries=all_timesteps_control_unitaries,
            noise=list_of_different_precomputed_noise.to(DEVICE),
        )
    )

    data_frame = pd.DataFrame(
        alpha_beta_gamma_sols.cpu().numpy(),
        index=noise_profiles_names,
        columns=COLUNMS_LABELS,
    )

    data_frame.index.name = "Noise_Type"
    data_frame.to_csv("./data_csv_files/ideal_CPMG_data_r2.csv")


if __name__ == "__main__":
    main()
