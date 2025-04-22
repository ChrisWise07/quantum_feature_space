import torch


def generate_gaussian_pulses(
    number_of_channels: int,
    time_range_values: torch.Tensor,
    pulse_parameters: torch.Tensor,
) -> torch.Tensor:
    """
    Generates a gaussian pulse for the given parameters.

    Args:
        number_of_channels (int):
            Number of channels for which to generate the pulses.
        time_range_values (torch.Tensor):
            Time range for which to generate the pulses.
            Expected Shape:
            (
                number_of_time_steps,
            )
        pulse_parameters (torch.Tensor): Parameters of the pulses.
            Expected Shape:
            (
                batch_size,
                number_of_pulses,
                number_of_channels * 3
            )
            Where the last dimension contains the amplitude, position
            and standard deviation of the pulse.

    Returns:
        torch.Tensor:
            Resulting gaussian pulse of shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            )

    """
    amplitude = torch.stack(
        [pulse_parameters[:, :, 0 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    position = torch.stack(
        [pulse_parameters[:, :, 1 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    std = torch.stack(
        [pulse_parameters[:, :, 2 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    time_range_expanded = time_range_values[None, None, :, None]
    amplitude_expanded = amplitude[:, :, None, :]
    position_expanded = position[:, :, None, :]
    std_expanded = std[:, :, None, :]

    signal = amplitude_expanded * torch.exp(
        -0.5 * ((time_range_expanded - position_expanded) / std_expanded) ** 2
    )

    return signal.sum(dim=1)


def generate_square_pulses(
    number_of_channels: int,
    time_range_values: torch.Tensor,
    pulse_parameters: torch.Tensor,
) -> torch.Tensor:
    """
    Generates a square pulse for the given parameters.

    Args:
        number_of_channels (int):
            Number of channels for which to generate the pulses.
        time_range_values (torch.Tensor):
            Time range for which to generate the pulses.
            Expected Shape:
            (
                number_of_time_steps,
            )
        pulse_parameters (torch.Tensor): Parameters of the pulses.
            Expected Shape:
            (
                batch_size,
                number_of_pulses,
                number_of_channels * 3
            )
            Where the last dimension contains the amplitude, position
            and width of the pulse.

    Returns:
        torch.Tensor:
            Resulting square pulse of shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            )

    """
    amplitude = torch.stack(
        [pulse_parameters[:, :, 0 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    center_position = torch.stack(
        [pulse_parameters[:, :, 1 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    width = torch.stack(
        [pulse_parameters[:, :, 2 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    time_range_expanded = time_range_values[None, None, :, None]
    amplitude_expanded = amplitude[:, :, None, :]
    center_position_expanded = center_position[:, :, None, :]
    width_expanded = width[:, :, None, :]

    start_of_pulse = center_position_expanded - 0.5 * width_expanded
    end_of_pulse = center_position_expanded + 0.5 * width_expanded

    signal = (
        amplitude_expanded
        * (time_range_expanded >= start_of_pulse)
        * (time_range_expanded < end_of_pulse)
    )

    return signal.sum(dim=1)


def generate_distorted_signal(
    original_signal: torch.Tensor,
    dft_matrix_of_transfer_func: torch.Tensor,
) -> torch.Tensor:
    """
    A function that generates a distorted signal from the given original
    signal.

    Args:
        original_signal (torch.Tensor):
            The original signal to distort. Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            ),

    Returns:
        torch.Tensor:
            The distorted signal. Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            )
    """
    (
        batch_size,
        number_of_time_steps,
        number_of_channels,
    ) = original_signal.shape

    x = original_signal.permute(0, 2, 1).to(dtype=torch.cfloat)

    x = torch.reshape(x, (batch_size, number_of_channels, number_of_time_steps, 1))

    return torch.real(torch.matmul(dft_matrix_of_transfer_func, x).squeeze()).permute(
        0, 2, 1
    )
