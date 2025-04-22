import torch
import random
import pandas as pd

from typing import List, Tuple
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


def generate_noise_profiles(
    total_num_profiles: int,
) -> Tuple[List[int], List[int], torch.Tensor]:
    s_ns_flags = []
    noise_types = []
    precomputed_noise = torch.empty(total_num_profiles, 1024, 2000, 2)

    for i in range(total_num_profiles // 6):
        alpha = random.uniform(0.7, 1.3)
        centre_point = random.uniform(0.1, 0.9)
        mu = random.uniform(0, 256)
        div_factor = random.randint(2, 15)

        one_on_f_noise_gen = OneOnFNoiseWithBumpNoiseGenerator(
            number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
            number_of_time_steps=NUM_TIME_STEPS,
            total_num_examples=TOTAL_PULSE_TRAINS,
            alpha=alpha,
            mu=0.0,
            threshold_freq=40.0,
            flat_value=1 / 40,
        )

        one_on_f_noise_ns_gen = OneOnFNoiseWithBumpNoiseGenerator(
            number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
            number_of_time_steps=NUM_TIME_STEPS,
            total_num_examples=TOTAL_PULSE_TRAINS,
            non_stationary_variant=True,
            alpha=alpha,
            mu=0.0,
            non_stationary_signal_centre_point=centre_point,
            threshold_freq=40.0,
            flat_value=1 / 40,
        )

        one_on_f_noise_w_bump_gen = OneOnFNoiseWithBumpNoiseGenerator(
            number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
            number_of_time_steps=NUM_TIME_STEPS,
            total_num_examples=TOTAL_PULSE_TRAINS,
            alpha=alpha,
            mu=mu,
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
            alpha=alpha,
            mu=mu,
            non_stationary_signal_centre_point=centre_point,
            sigma=5.0,
            height=0.5,
            threshold_freq=40.0,
            flat_value=1 / 40,
        )

        coloured_noise_gen = ColouredGaussianNoiseGenerator(
            number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
            number_of_time_steps=NUM_TIME_STEPS,
            total_num_examples=TOTAL_PULSE_TRAINS,
            division_factor=div_factor,
        )

        coloured_noise_ns_gen = ColouredGaussianNoiseGenerator(
            number_of_noise_realisations=NUM_NOISE_REALIZATIONS,
            number_of_time_steps=NUM_TIME_STEPS,
            total_num_examples=TOTAL_PULSE_TRAINS,
            division_factor=div_factor,
            non_stationary_variant=True,
        )

        s_ns_flags.extend([0, 1, 0, 1, 0, 1])
        noise_types.extend([0, 0, 1, 1, 2, 2])

        # divide by constant values so signal energy is all roughly the same
        precomputed_noise[i * 6 : i * 6 + 6] = 4 * torch.concat(
            (
                CombinedNoiseGenerator(
                    x_noise_generator=one_on_f_noise_gen
                ).precomputed_noise_combined_noise
                / 2.75,
                CombinedNoiseGenerator(
                    x_noise_generator=one_on_f_noise_ns_gen
                ).precomputed_noise_combined_noise,
                CombinedNoiseGenerator(
                    x_noise_generator=one_on_f_noise_w_bump_gen
                ).precomputed_noise_combined_noise
                / 3.25,
                CombinedNoiseGenerator(
                    x_noise_generator=one_on_f_noise_w_bump_ns_gen
                ).precomputed_noise_combined_noise
                / 1.17,
                CombinedNoiseGenerator(
                    x_noise_generator=coloured_noise_gen
                ).precomputed_noise_combined_noise
                / 10.95,
                CombinedNoiseGenerator(
                    x_noise_generator=coloured_noise_ns_gen
                ).precomputed_noise_combined_noise
                / 6.27,
            )
        )

    return s_ns_flags, noise_types, precomputed_noise


def main() -> None:
    batch_size = 7
    total_num_profiles = 6 * 100

    s_ns_flags, noise_types, precomputed_noise = generate_noise_profiles(
        total_num_profiles=total_num_profiles
    )

    num_batches = total_num_profiles // batch_size + 1

    all_timesteps_control_unitaries = (
        STANDARD_QUBIT_SIM.compute_all_timesteps_control_unitaries(
            control_pulse_time_series=IDEAL_CPMG_CONTROL_PULSES,
        )
    )

    alpha_beta_gamma_sols = torch.empty(total_num_profiles, 9)

    for i in range(num_batches):
        print(f"Batch {i + 1}/{num_batches}")
        start = i * batch_size
        end = (i + 1) * batch_size

        alpha_beta_gamma_sols[start:end] = (
            STANDARD_QUBIT_SIM.compute_alpha_beta_gamma_sols_for_control_and_noise(
                all_timesteps_control_unitaries=all_timesteps_control_unitaries,
                noise=precomputed_noise[start:end].to(DEVICE),
            )
        )

        alpha_beta_gamma_sols[start:end] = alpha_beta_gamma_sols[start:end].to("cpu")

    data_frame = pd.DataFrame(
        alpha_beta_gamma_sols.cpu().numpy(),
        columns=COLUNMS_LABELS,
    )

    data_frame["stationarity"] = s_ns_flags
    data_frame["noise_type"] = noise_types
    data_frame.to_csv("./data_csv_files/ideal_CPMG_data_for_dt.csv")


if __name__ == "__main__":
    main()
