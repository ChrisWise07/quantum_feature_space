import torch
import math

torch.manual_seed(0)

PI = math.pi
EPSILON = 1e-8
QUBIT_ENERGY_GAP = 12.0
NUMBER_OF_EXPECTATIONS_SINGLE_QUBIT = 3 * 6**1
NUMBER_OF_EXPECTATIONS_TWO_QUBITS = 15 * 6**2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA_I = torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat, device=DEVICE)
SIGMA_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=DEVICE)
SIGMA_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=DEVICE)
SIGMA_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=DEVICE)

# Tensor has shape (1, 3, 1, 2, 2) for batched matrix multiplication
COMBINED_SIGMA_TENSOR = (
    torch.stack([SIGMA_X, SIGMA_Y, SIGMA_Z], dim=0).unsqueeze(0)
).to(DEVICE)

# Tensor has shape (1, 3, 1, 4, 4) for batched matrix multiplication
COMBINED_SIGMA_TENSOR_TWO_QUBITS = (
    torch.stack(
        [
            torch.kron(SIGMA_ONE, SIGMA_TWO)
            for SIGMA_ONE in [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]
            for SIGMA_TWO in [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]
        ],
        dim=0,
    )[1:].unsqueeze(0)
).to(DEVICE)

LIST_OF_PAULI_EIGENSTATES = torch.tensor(
    [
        [[1 / 2, 1 / 2], [1 / 2, 1 / 2]],
        [[1 / 2, -1 / 2], [-1 / 2, 1 / 2]],
        [[1 / 2, -1j / 2], [1j / 2, 1 / 2]],
        [[1 / 2, 1j / 2], [-1j / 2, 1 / 2]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
    ],
    dtype=torch.cfloat,
    device=DEVICE,
)

LIST_OF_PAULI_EIGENSTATES_TWO_QUBITS = torch.stack(
    [
        torch.kron(STATE_ONE, STATE_TWO)
        for STATE_ONE in LIST_OF_PAULI_EIGENSTATES
        for STATE_TWO in LIST_OF_PAULI_EIGENSTATES
    ],
    dim=0,
).to(DEVICE)


UNIVERSAL_GATE_SET_SINGLE_QUBIT = {
    "I": torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat, device=DEVICE),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=DEVICE),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=DEVICE),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=DEVICE),
    "H": torch.tensor(
        [
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), -1 / math.sqrt(2)],
        ],
        dtype=torch.cfloat,
        device=DEVICE,
    ),
    "R_X_PI/4": torch.tensor(
        [
            [math.cos(math.pi / 8), -1j * math.sin(math.pi / 8)],
            [-1j * math.sin(math.pi / 8), math.cos(math.pi / 8)],
        ],
        dtype=torch.cfloat,
        device=DEVICE,
    ),
}

LAMBDA_MATRIX_FOR_CHI_MATRIX = (
    1
    / 2
    * torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, -1, 0],
            [1, 0, 0, -1],
        ],
        dtype=torch.cfloat,
        device=DEVICE,
    ).unsqueeze(0)
)


IDEAL_PROCESS_MATRICES = {
    "I": torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "X": torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "Y": torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "Z": torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "H": torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5],
        ],
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "R_X_PI/4": torch.tensor(
        [
            [0.8536 + 0.0j, 0.0 + 0.3536j, 0.0 + 0.0j, 0.0 - 0.0j],
            [0.0 - 0.3536j, 0.1464 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        device=DEVICE,
        dtype=torch.cfloat,
    ),
}

IDEAL_EXPECTATIONS_FOR_SINGLE_QUBIT = {
    "I": torch.tensor(
        [
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ]
    ),
    "X": torch.tensor(
        [
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
            ]
        ]
    ),
    "Y": torch.tensor(
        [
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
            ]
        ]
    ),
    "Z": torch.tensor(
        [
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ]
    ),
    "H": torch.tensor(
        [
            [
                [-0.0, -0.0, -0.0, -0.0, 1.0, -1.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        device="cuda:0",
    ),
    "R_X_PI/4": torch.tensor(
        [
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-0.0, -0.0, 0.707107, -0.707107, -0.707107, 0.707107],
                [0.0, 0.0, 0.707107, -0.707107, 0.707107, -0.707107],
            ]
        ],
        device="cuda:0",
    ),
}
