import torch
from typing import Union, Optional


def generate_time_domain_signal_for_noise(
    time_range: torch.Tensor,
    total_time: Union[float, int],
    peak_position: int = 0.5,
) -> torch.Tensor:
    """
    Generate a time domain signal for to be used in the generation of
    non-stationary noise. See:
    https://www.nature.com/articles/s41597-022-01639-1
    for details.

    Args:
        time_range (torch.Tensor):
            The time range for which to generate the signal. Shape:
            (
                number_of_time_steps,
            )
        total_time (Union[float, int]):
            Total time for which the system is simulated.
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.

    Returns:
        torch.Tensor:
            The time domain signal. Shape:
            (
                number_of_time_steps,
            )
    """
    return 1 - (torch.abs(time_range - peak_position * total_time) * 2)


def generate_spectral_density_of_one_on_f_noise_with_bump(
    frequencies: torch.Tensor,
    alpha: float = 1,
    mu: float = 30,
    sigma: float = 5,
    height: float = 0.5,
    threshold_freq: Optional[float] = None,
    flat_value: float = 1 / 16,
) -> torch.Tensor:
    """
    Generate a spectral density that has 1/f PSD with a Gaussian bump,
    where the location, width of the bump, and the flat region beyond a
    certain frequency can be parameterized.

    Args:
        frequencies:
            The frequencies for which to generate the spectral density.
        alpha: Controls the slope of the 1/f noise.
        mu: The mean of the Gaussian bump in the frequency domain.
        sigma: The standard deviation   of the Gaussian bump.
        height: The height of the Gaussian bump.
        threshold_freq:
            The frequency threshold beyond which the spectral density is
            flat.
        flat_value:
            The value of the spectral density beyond the threshold
            frequency.

    Returns:
        torch.Tensor: The generated spectral density.
    """
    positive_frequencies = frequencies[frequencies >= 0]

    base_one_on_f_frequencies = 1 / (positive_frequencies + 1) ** alpha

    if threshold_freq is None:
        return base_one_on_f_frequencies

    one_on_f_frequencies = base_one_on_f_frequencies * (
        positive_frequencies <= threshold_freq
    )

    flat_values = flat_value * (positive_frequencies > threshold_freq)

    gaussian_bump = height * torch.exp(
        -((positive_frequencies - mu) ** 2) / (2 * sigma**2)
    )

    return one_on_f_frequencies + flat_values + gaussian_bump


def generate_sqrt_scaled_spectral_density(
    total_time: Union[float, int],
    spectral_density: torch.Tensor,
    number_of_time_steps: int,
) -> torch.Tensor:
    """
    A function which generates a spectral density that has been scaled
    and then square rooted. This spectral density can be used to
    generate noise with 1/f PSD.

    Args:
        total_time (Union[float, int]):
            Total time for which the system is simulated.
        spectral_density (torch.Tensor):
            The spectral density to scale and square root. Shape:
            (
                number_of_time_steps,
            )
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        number_of_noise_realisations (int):
            Number of noise realisations to generate.

    Returns:
        torch.Tensor:
            Tensor of values which can be used to generate noise with
            1/f PSD. Shape:
            (
                number_of_time_steps // 2
            )
    """
    return torch.sqrt(spectral_density * (number_of_time_steps**2) / total_time)


def gerenate_noise_time_series_with_one_on_f_noise(
    sqrt_scaled_spectral_density: torch.Tensor,
    number_of_time_steps: int,
    number_of_noise_realisations: int,
    total_num_examples: int,
) -> torch.Tensor:
    """
    A function that generates a noise time domain signal.

    Args:
        sqrt_scaled_spectral_density (torch.Tensor):
            The spectral density to scale and square root. Shape:
            (
                number_of_time_steps // 2,
            )
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        number_of_noise_realisations (int):
            Number of noise realisations to generate.
        batch_size (int): Number of systems to simulate.

    Returns:
        torch.Tensor:
            The generated noise signal. Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_noise_realisations,

            )
    """
    sqrt_scaled_spectral_density_expanded = sqrt_scaled_spectral_density[
        None, None, ...
    ]

    spectral_density_randomised_phases = (
        sqrt_scaled_spectral_density_expanded
        * torch.exp(
            2
            * torch.pi
            * 1j
            * torch.rand(
                (
                    total_num_examples,
                    number_of_noise_realisations,
                    number_of_time_steps // 2,
                ),
            )
        )
    )

    noise = torch.fft.ifft(
        torch.concatenate(
            (
                spectral_density_randomised_phases,
                spectral_density_randomised_phases.conj().flip(dims=[-1]),
            ),
            dim=-1,
        ),
        dim=-1,
    ).real

    return noise.permute(0, 2, 1)


def generate_colour_filter_for_noise(
    number_of_time_steps: int,
    division_factor: int = 4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a colour filter for coloured noise. This is a filter that
    is used to generate coloured noise. See:
    https://www.nature.com/articles/s41597-022-01639-1
    for details.

    Args:
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        division_factor (int, optional):
            The division factor for the colour filter. Defaults to 4.
            This will control the frequency of the noise.

    Returns:
        torch.Tensor:
            The colour filter. Shape:
            (
                number_of_time_steps // division_factor,
            )
    """
    return torch.ones((number_of_time_steps // division_factor), device=device)


def generate_coloured_gaussian_noise(
    total_num_examples: int,
    colour_filter: torch.Tensor,
    number_of_time_steps: int,
    number_of_noise_realisations: int,
    division_factor: int = 4,
    device: Optional[torch.device] = None,
):
    """
    Generate non-stationary coloured Gaussian noise with specified
    time domain signal and colour filter. Can implement noise profile 3
    and 4 See: https://www.nature.com/articles/s41597-022-01639-1 for
    details.

    Args:
        batch_size (int):
            Number of systems to simulate.
        time_domain_signal (torch.Tensor):
            The time domain signal. Shape:
            (
                number_of_time_steps,
            )
        colour_filter (torch.Tensor):
            The colour filter. Shape:
            (
                number_of_time_steps // division_factor,
            )
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        number_of_noise_realisations (int):
            Number of noise realisations to generate.
        division_factor (int, optional):
            The division factor for the colour filter. Defaults to 4.
            This will control the frequency of the noise.
    """
    random_numbers = torch.randn(
        (
            total_num_examples * number_of_noise_realisations,
            1,
            number_of_time_steps + (number_of_time_steps // division_factor) - 1,
        ),
        device=device,
    )

    colour_filter_expanded = colour_filter[None, None, ...]

    noise = torch.nn.functional.conv1d(
        random_numbers, colour_filter_expanded, padding="valid"
    )

    noise = noise.view(total_num_examples, number_of_noise_realisations, -1)
    return noise.permute(0, 2, 1)
