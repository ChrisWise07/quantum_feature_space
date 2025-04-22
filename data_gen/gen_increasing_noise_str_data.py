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

one_on_f_noise_generator = OneOnFNoiseWithBumpNoiseGenerator(
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

coloured_noise_ns = ColouredGaussianNoiseGenerator(
    number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
    number_of_time_steps=NUM_TIME_STEPS,
    total_num_examples=TOTAL_PULSE_TRAINS,
    non_stationary_variant=True,
)

one_on_f_combined_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=one_on_f_noise_generator,
)

one_on_f_precomputed_noise_scaled = (
    4 / 3.25
) * one_on_f_combined_noise_generator.precomputed_noise_combined_noise

coloured_noise_ns_combined_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=coloured_noise_ns,
)

coloured_noise_ns_precomputed_noise_scaled = (
    4 / 6.27
) * coloured_noise_ns_combined_noise_generator.precomputed_noise_combined_noise

noise_strengths = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

one_on_f_increasing_str_precomputed_noise = torch.concat(
    [s * one_on_f_precomputed_noise_scaled for s in noise_strengths]
)

coloured_noise_ns_increasing_str_precomputed_noise = torch.concat(
    [s * coloured_noise_ns_precomputed_noise_scaled for s in noise_strengths]
)


def main() -> None:
    all_timesteps_control_unitaries = (
        STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
            control_pulse_time_series=IDEAL_CPMG_CONTROL_PULSES,
        )
    )

    for precomputed_noise, name in zip(
        [
            one_on_f_increasing_str_precomputed_noise,
            coloured_noise_ns_increasing_str_precomputed_noise,
        ],
        ["one_on_f", "coloured_noise_ns"],
    ):
        alpha_beta_gamma_sols = (
            STANDARD_QUBIT_SIM.compute_alpha_beta_gamma_sols_for_control_and_noise(
                all_timesteps_control_unitaries=all_timesteps_control_unitaries,
                noise=precomputed_noise.to(DEVICE),
            )
        )

        data_frame = pd.DataFrame(
            alpha_beta_gamma_sols.cpu().numpy(),
            index=noise_strengths,
            columns=COLUNMS_LABELS,
        )

        data_frame.index.name = "Noise Strength"
        data_frame.to_csv(f"./data_csv_files/data_for_increasing_noise_{name}.csv")


if __name__ == "__main__":
    main()
