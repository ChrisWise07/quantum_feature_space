import torch
import pandas as pd

from qubit_sim.constants import DEVICE

from qubit_sim.noise_gen_classes import (
    OneOnFNoiseWithBumpNoiseGenerator,
    ColouredGaussianNoiseGenerator,
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

one_on_f_noise_gen = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    alpha=1.0,
    mu=0.0,
    threshold_freq=40.0,
    flat_value=1 / 40,
)

one_on_f_noise_ns_gen = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    non_stationary_variant=True,
    alpha=1.0,
    mu=0.0,
    threshold_freq=40.0,
    flat_value=1 / 40,
)

one_on_f_noise_w_bump_gen = OneOnFNoiseWithBumpNoiseGenerator(
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

one_on_f_noise_w_bump_ns_gen = OneOnFNoiseWithBumpNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    non_stationary_variant=True,
    alpha=1.0,
    mu=30.0,
    sigma=5.0,
    height=0.5,
    threshold_freq=40.0,
    flat_value=1 / 40,
)

coloured_noise_gen = ColouredGaussianNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
)

coloured_noise_ns = ColouredGaussianNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    non_stationary_variant=True,
)

one_on_f_noise_comb_gen = CombinedNoiseGenerator(x_noise_generator=one_on_f_noise_gen)

one_on_f_noise_comb_ns_gen = CombinedNoiseGenerator(
    x_noise_generator=one_on_f_noise_ns_gen
)

one_on_f_noise_w_bump_comb_gen = CombinedNoiseGenerator(
    x_noise_generator=one_on_f_noise_w_bump_gen
)

one_on_f_noise_w_bump_comb_ns_gen = CombinedNoiseGenerator(
    x_noise_generator=one_on_f_noise_w_bump_ns_gen
)

coloured_noise_comb_gen = CombinedNoiseGenerator(x_noise_generator=coloured_noise_gen)

coloured_noise_comb_ns_gen = CombinedNoiseGenerator(x_noise_generator=coloured_noise_ns)

# divide by constant values so signal energy is all roughly the same
list_of_different_precomputed_noise = 4 * torch.concat(
    (
        one_on_f_noise_comb_gen.precomputed_noise_combined_noise / 2.75,
        one_on_f_noise_comb_ns_gen.precomputed_noise_combined_noise,
        one_on_f_noise_w_bump_comb_gen.precomputed_noise_combined_noise / 3.25,
        one_on_f_noise_w_bump_comb_ns_gen.precomputed_noise_combined_noise / 1.17,
        coloured_noise_comb_gen.precomputed_noise_combined_noise / 10.95,
        coloured_noise_comb_ns_gen.precomputed_noise_combined_noise / 6.27,
    )
)

noise_profiles_names = [
    "one_on_f_noise",
    "one_on_f_noise_ns",
    "one_on_f_noise_w_bump",
    "one_on_f_noise_w_bump_ns",
    "coloured_noise",
    "coloured_noise_ns",
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
    data_frame.to_csv("./data_csv_files/ideal_CPMG_data.csv")


if __name__ == "__main__":
    main()
