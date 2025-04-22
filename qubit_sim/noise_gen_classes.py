import torch

from dataclasses import dataclass
from typing import Callable, Optional, Union

from qubit_sim.noise_gen_funcs import (
    generate_time_domain_signal_for_noise,
    generate_spectral_density_of_one_on_f_noise_with_bump,
    generate_sqrt_scaled_spectral_density,
    gerenate_noise_time_series_with_one_on_f_noise,
    generate_colour_filter_for_noise,
    generate_coloured_gaussian_noise,
)


@dataclass
class BaseNoiseGenerator:
    """
    Base class for noise generators.

    Attributes:
        number_of_noise_realisations (int):
            Number of noise realisations for a single example.
        number_of_time_steps (int):
            Number of time steps for a control pulse, i.e. the number of
            values in the control pulse sequence.
        total_time (float):
            The total time for the noise (default 1.0).
        g (float):
            The strength of the noise (default 0.2).
    """

    number_of_noise_realisations: int = 2000
    number_of_time_steps: int = 1024
    total_time: float = 1.0
    total_num_examples: int = 50
    non_stationary_variant: bool = False
    non_stationary_signal_centre_point: float = 0.5
    precomputed_noise: Optional[torch.Tensor] = None
    noise_generator_func: Optional[Callable] = None
    noise_generator_func_kwargs: Optional[dict] = None
    device: Optional[torch.device] = None
    precompute_noise: bool = True

    def __post_init__(self):
        if self.non_stationary_variant:
            time_step = self.total_time / self.number_of_time_steps

            time_range = torch.linspace(
                0,
                self.total_time - time_step,
                self.number_of_time_steps,
            )

            self.time_domain_signal = self.generate_signal_for_non_stationary_noise(
                time_range=time_range,
                total_time=self.total_time,
            )

    def generate_signal_for_non_stationary_noise(
        self,
        time_range: torch.Tensor,
        total_time: float,
    ) -> torch.Tensor:
        """
        Generate the signal for non-stationary noise.

        Args:
            time_range (torch.Tensor):
                The time range for the signal.
            total_time (float):
                The total time for the noise.

        Returns:
            signal (torch.Tensor):
                The generated signal. Expected shape:
                (
                    number_of_time_steps,
                )
        """
        return generate_time_domain_signal_for_noise(
            time_range=time_range,
            total_time=total_time,
            peak_position=self.non_stationary_signal_centre_point,
        )

    def generate_noise_instance(self, total_num_examples: int) -> torch.Tensor:
        """
        Generate noise.

        Args:
            total_num_examples (int):
                The batch size.

        Returns:
            noise (torch.Tensor):
                The generated noise. Expected shape:
                (
                    total_num_examples,
                    number_of_time_steps,
                )
        """
        with torch.no_grad():
            noise = self.noise_generator_func(
                **self.noise_generator_func_kwargs,
                total_num_examples=total_num_examples
            )

            if self.non_stationary_variant:
                return noise * self.time_domain_signal.unsqueeze(0).unsqueeze(-1)

            return noise


@dataclass
class OneOnFNoiseWithBumpNoiseGenerator(BaseNoiseGenerator):
    """
    Class to generate 1/f noise.

    Attributes:
        alpha (float):
            The alpha parameter for the 1/f noise.
    """

    alpha: float = 1.0
    mu: float = 0.0
    sigma: float = 0.2
    height: float = 0.1
    threshold_freq: float = 15
    flat_value: float = 1 / 16

    def __post_init__(self):
        super().__post_init__()

        frequencies = torch.fft.fftfreq(
            n=self.number_of_time_steps,
            d=self.total_time / self.number_of_time_steps,
        )

        spectral_density = generate_spectral_density_of_one_on_f_noise_with_bump(
            frequencies=frequencies,
            alpha=self.alpha,
            mu=self.mu,
            sigma=self.sigma,
            height=self.height,
            threshold_freq=self.threshold_freq,
            flat_value=self.flat_value,
        )

        self.sqrt_scaled_spectral_density = generate_sqrt_scaled_spectral_density(
            total_time=self.total_time,
            spectral_density=spectral_density,
            number_of_time_steps=self.number_of_time_steps,
        )

        self.noise_generator_func_kwargs = {
            "sqrt_scaled_spectral_density": self.sqrt_scaled_spectral_density,
            "number_of_noise_realisations": self.number_of_noise_realisations,
            "number_of_time_steps": self.number_of_time_steps,
        }

        self.noise_generator_func = gerenate_noise_time_series_with_one_on_f_noise

        self.precomputed_noise = self.generate_noise_instance(
            total_num_examples=self.total_num_examples
        )


@dataclass
class ColouredGaussianNoiseGenerator(BaseNoiseGenerator):
    """
    Class to generate coloured Gaussian noise.

    Attributes:
        number_of_noise_realisations (int):
            Number of noise realisations for a single example.
        number_of_time_steps (int):
            Number of time steps for a control pulse, i.e. the number of
            values in the control pulse sequence.
    """

    division_factor: int = 4

    def __post_init__(self):
        super().__post_init__()

        self.colour_filter = generate_colour_filter_for_noise(
            number_of_time_steps=self.number_of_time_steps,
            division_factor=self.division_factor,
            device=self.device,
        )

        self.noise_generator_func_kwargs = {
            "colour_filter": self.colour_filter,
            "number_of_time_steps": self.number_of_time_steps,
            "number_of_noise_realisations": self.number_of_noise_realisations,
            "division_factor": self.division_factor,
            "device": self.device,
        }

        self.noise_generator_func = generate_coloured_gaussian_noise

        if self.precompute_noise:
            self.precomputed_noise = self.generate_noise_instance(
                total_num_examples=self.total_num_examples
            )


@dataclass
class RandomTelegraphNoiseGenerator(BaseNoiseGenerator):
    """
    Class to generate random telegraph noise.
    """

    transition_probs_to_zero_state: torch.Tensor = torch.tensor([0.5, 0.5])

    def __post_init__(self):
        with torch.no_grad():
            self.precomputed_noise = self.generate_noise_instance(
                total_num_examples=self.total_num_examples
            )

    def generate_noise_instance(self, total_num_examples: int) -> torch.Tensor:
        """
        Generate random telegraph noise.

        Args:
            total_num_examples (int):
                The batch size.

        Returns:
            noise (torch.Tensor):
                The generated noise. Expected shape:
                (
                    total_num_examples,
                    number_of_time_steps,
                )
        """
        initial_state = torch.where(
            torch.rand(total_num_examples, self.number_of_noise_realisations) < 0.5,
            0,
            1,
        )

        noise = torch.empty(
            total_num_examples,
            self.number_of_time_steps,
            self.number_of_noise_realisations,
            dtype=torch.int,
        )

        noise[:, 0, :] = initial_state

        for j in range(1, self.number_of_time_steps):
            probs = self.transition_probs_to_zero_state[noise[:, j - 1, :]]

            next_states = torch.where(
                torch.rand(
                    total_num_examples,
                    self.number_of_noise_realisations,
                )
                < probs,
                0,
                1,
            )

            noise[:, j, :] = next_states

        noise = torch.where(noise == 0, -1, 1)

        return noise.to(torch.float)


@dataclass
class CombinedNoiseGenerator:
    """
    A class the combines two noise generators, one for x and one for z
    axis.

    Attributes:
        x_noise_generator (Class):
            The noise generator for the x axis.
        z_noise_generator (Class):
            The noise generator for the z axis.
    """

    x_noise_generator: Optional[
        Union[
            OneOnFNoiseWithBumpNoiseGenerator,
            ColouredGaussianNoiseGenerator,
            RandomTelegraphNoiseGenerator,
        ]
    ] = None

    z_noise_generator: Optional[
        Union[
            OneOnFNoiseWithBumpNoiseGenerator,
            ColouredGaussianNoiseGenerator,
            RandomTelegraphNoiseGenerator,
        ]
    ] = None

    def __post_init__(self):
        if not self.x_noise_generator and not self.z_noise_generator:
            raise ValueError(
                "x_noise_generator and z_noise_generator cannot be None at the same time."
            )

        precompute_noise = (
            self.x_noise_generator.precompute_noise
            if self.x_noise_generator is not None
            else self.z_noise_generator.precompute_noise
        )

        if precompute_noise:
            if self.z_noise_generator is None:
                self.precomputed_noise_x_noise = (
                    self.x_noise_generator.precomputed_noise
                )
                self.precomputed_noise_z_noise = torch.abs(
                    self.precomputed_noise_x_noise
                )

            if self.x_noise_generator is None:
                self.precomputed_noise_z_noise = (
                    self.z_noise_generator.precomputed_noise
                )
                self.precomputed_noise_x_noise = torch.abs(
                    self.precomputed_noise_z_noise
                )

            self.precomputed_noise_combined_noise = torch.stack(
                (
                    self.precomputed_noise_x_noise,
                    self.precomputed_noise_z_noise,
                ),
                dim=-1,
            )

    def generate_new_noise_instance(self, total_num_examples: int) -> torch.Tensor:
        """
        Generates the noise for the x and z axis and combines them.

        Args:
            total_num_examples (int):
                The batch size.

        Returns:
            noise (torch.Tensor):
                The generated noise. Expected shape:
                (
                    total_num_examples,
                    number_of_time_steps,
                    2,
                )
        """
        if self.z_noise_generator is None:
            noise_x = self.x_noise_generator.generate_noise_instance(
                total_num_examples=total_num_examples
            )

            noise_z = torch.abs(noise_x)

            return torch.stack(
                (
                    noise_x,
                    noise_z,
                ),
                dim=-1,
            )

        if self.x_noise_generator is None:
            noise_z = self.z_noise_generator.generate_noise_instance(
                total_num_examples=total_num_examples
            )

            noise_x = torch.abs(noise_z)

            return torch.stack(
                (
                    noise_x,
                    noise_z,
                ),
                dim=-1,
            )

        noise_x = self.x_noise_generator.generate_noise_instance(
            total_num_examples=total_num_examples
        )

        noise_z = self.z_noise_generator.generate_noise_instance(
            total_num_examples=total_num_examples
        )

        return torch.stack(
            (
                noise_x,
                noise_z,
            ),
            dim=-1,
        )
