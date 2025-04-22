import torch

from qubit_sim.qubit_sim_class import QubitSimulator
from qubit_sim.constants import SIGMA_X, SIGMA_Z, PI, DEVICE
from qubit_sim.control_pulse_funcs import generate_gaussian_pulses

MAX_AMP = 1.0
TOTAL_TIME = 1.0
NUM_TIME_STEPS = 1024
QUBIT_ENERGY_GAP = 12
TOTAL_PULSE_TRAINS = 1
NUM_CONTROL_CHANNELS = 1
NUM_PULSES_PER_TRAIN = 5
NUM_NOISE_REALIZATIONS = 2000
IDEAL_CPMG_PULSE_SCALE = 1 / 96
TIME_STEP = TOTAL_TIME / NUM_TIME_STEPS
CONTROL_DYNAMIC_OPERATORS = 0.5 * SIGMA_X
NOISE_DYNAMIC_OPERATORS = torch.stack((0.5 * SIGMA_X, 0.5 * SIGMA_Z), dim=0)
IDEAL_CPMG_STD_VALUE = TOTAL_TIME / NUM_PULSES_PER_TRAIN * IDEAL_CPMG_PULSE_SCALE
TIME_STEP_VALUES = torch.linspace(0, TOTAL_TIME - TIME_STEP, NUM_TIME_STEPS).to(DEVICE)

STANDARD_QUBIT_SIM = QubitSimulator(
    delta_t=TOTAL_TIME / NUM_TIME_STEPS,
    max_amp=MAX_AMP,
    noise_dynamic_operators=NOISE_DYNAMIC_OPERATORS,
    control_static_operators=None,
    control_dynamic_operators=CONTROL_DYNAMIC_OPERATORS,
)

COLUNMS_LABELS = [
    "Vx_alpha",
    "Vx_beta",
    "Vx_gamma",
    "Vy_alpha",
    "Vy_beta",
    "Vy_gamma",
    "Vz_alpha",
    "Vz_beta",
    "Vz_gamma",
]

gaussian_centre_positions = (
    (torch.arange(1, NUM_PULSES_PER_TRAIN + 1) - 0.5)
    / NUM_PULSES_PER_TRAIN
    * TOTAL_TIME
)

gaussian_centre_positions = gaussian_centre_positions[None, :, None]
gaussian_amplitudes = torch.full((1, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS), PI)

gaussian_std_values = torch.full(
    (1, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS), IDEAL_CPMG_STD_VALUE
)

gaussian_pulse_parameters = torch.stack(
    (gaussian_amplitudes, gaussian_centre_positions, gaussian_std_values), dim=-1
)

pulse_parameters = gaussian_pulse_parameters.reshape(
    1, NUM_PULSES_PER_TRAIN, NUM_CONTROL_CHANNELS * 3
)

IDEAL_CPMG_CONTROL_PULSES = generate_gaussian_pulses(
    number_of_channels=NUM_CONTROL_CHANNELS,
    time_range_values=TIME_STEP_VALUES,
    pulse_parameters=pulse_parameters.to(DEVICE),
)
