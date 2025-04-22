import os
import time
import torch
import pickle
import numpy as np
import multiprocessing

from natsort import natsorted
from typing import Dict, Tuple
from qubit_sim.constants import SIGMA_X, SIGMA_Y, SIGMA_Z
from qubit_sim.utils import calculate_quantum_features_from_expectation_values

PATH_TO_DATA_FOLDER = "./G_1q_XY_XZ_N3N6_comb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_pickle(
    filename_dataset_path_tuple: Tuple[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        filename: Name of the file to load.
        path_to_dataset: Path to the dataset zip file.

    Returns:
        Tuple of numpy arrays containing (in order):
            pulses: Array of pulses.
            VO: Array of Vo operators.
            expectations: Array of expectation values.
            control_unitaries: Array of control unitaries.
            pulse_parameters: Array of pulse parameters.

    """
    filename, path_to_dataset = filename_dataset_path_tuple

    with open(f"{path_to_dataset}/{filename}", "rb") as f:
        data = pickle.load(f)

    return (
        data["Vo_operator"],
        data["expectations"],
        data["U0"],
    )


def load_qdataset(
    path_to_dataset: str, num_examples: int, use_pulse_parameters: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Load the dataset of pulses, Vo operators, expectation values,
    control unitaries, and pulse parameters.

    Args:
        path_to_dataset: Path to the dataset
        num_examples: Number of examples to load.

    Returns:
        Dictionary containing:
            Vx: torch.Tensor of X Vo operators.
            Vy: torch.Tensor of Y Vo operators.
            Vz: torch.Tensor of Z Vo operators.
            expectations: torch.Tensor of expectation values.
            control_unitaries: torch.Tensor of control unitaries.
            if use_pulse_parameters:
                pulse_parameters: torch.Tensor of pulse parameters.
            else:
                pulses: torch.Tensor of pulses.
    """
    filenames = natsorted(os.listdir(path_to_dataset))[:num_examples]

    with multiprocessing.Pool() as pool:
        func = pool.map_async(
            load_data_from_pickle,
            [[filename, path_to_dataset] for filename in filenames],
        )

        Vo, expectations, control_unitaries = zip(*func.get())

        Vo = np.array(Vo)
        expectations = np.array(expectations).squeeze()
        control_unitaries = np.array(control_unitaries).squeeze()

        return {
            "Vo": torch.from_numpy(Vo.squeeze()),
            "expectations": torch.from_numpy(expectations),
            "control_unitaries": torch.from_numpy(control_unitaries)[:, -1],
        }


def test_Vo_construction_from_expectation_values_for_batch():
    batch_size = 3
    torch.set_printoptions(precision=6, sci_mode=False)

    data = load_qdataset(
        path_to_dataset=PATH_TO_DATA_FOLDER,
        num_examples=batch_size,
    )

    Qx = SIGMA_X @ data["Vo"][:, 0].to(DEVICE)
    Qy = SIGMA_Y @ data["Vo"][:, 1].to(DEVICE)
    Qz = SIGMA_Z @ data["Vo"][:, 2].to(DEVICE)
    qdq_dag_ops = torch.stack((Qx, Qy, Qz), dim=1)
    alpha_sols = torch.real(qdq_dag_ops[:, :, 0, 1])
    beta_sols = -torch.imag(qdq_dag_ops[:, :, 0, 1])
    gamma_sols = torch.real(qdq_dag_ops[:, :, 0, 0])
    stacked = torch.stack((alpha_sols, beta_sols, gamma_sols), dim=2)

    expectations = (
        data["expectations"].view(batch_size, 6, 3).permute(0, 2, 1).to(DEVICE)
    )

    control_unitaries = data["control_unitaries"].to(DEVICE)

    timings = []
    print("Timing Vo construction")

    for i in range(10):
        start = time.time()

        feature_space = calculate_quantum_features_from_expectation_values(
            expectation_values=expectations,
            control_unitaries=control_unitaries,
        )

        end = time.time()
        timings.append(end - start)

    print(f"Mean time (ms): {np.mean(timings[1:]) * 1000:.2f}")
    print(f"Std time (ms): {np.std(timings[1:]) * 1000:.2f}")
    print(torch.mean(torch.abs(feature_space - stacked)))
    print(torch.max(torch.abs(feature_space - stacked)))


if __name__ == "__main__":
    test_Vo_construction_from_expectation_values_for_batch()
