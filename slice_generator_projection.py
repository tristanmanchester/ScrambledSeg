"""
Synthetic X-ray Computed Tomography (XCT) Slice Generator (Multi-Label)

This script generates synthetic 2D XCT slices that simulate microstructural features
of composite battery cathodes, specifically NMC particles in a solid electrolyte matrix.
It also generates a corresponding multi-label segmentation image.

Includes an optional forward projection/reconstruction step to simulate
tomographic acquisition artifacts.

Refactored to use a central configuration dictionary for easier parameter management.

Label mapping for the segmentation image:
- 0: Out of Reconstruction Volume
- 1: Background (Outer Ring)
- 2: Electrolyte (Inner Circle)
- 3: Cathode Particles
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeAlias, TypeVar
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from random import uniform, randint
from copy import deepcopy
from tifffile import imwrite
from pathlib import Path

from skimage.transform import radon, iradon
from scipy.ndimage import gaussian_filter1d

LABEL_OUT_OF_RECONSTRUCTION = 0
LABEL_BACKGROUND = 1
LABEL_ELECTROLYTE = 2
LABEL_CATHODE = 3

ConfigScalar: TypeAlias = int | float | bool | str | Path
ConfigValue: TypeAlias = ConfigScalar | tuple[int, int] | list["ConfigValue"] | dict[str, "ConfigValue"]
ConfigDict: TypeAlias = dict[str, ConfigValue]
DataclassT = TypeVar("DataclassT")


@dataclass
class ImageParameters:
    size: int
    outer_circle_size: int
    inner_circle_size: int
    base_grey: int
    inner_circle_grey: int
    outer_circle_grey: int
    particle_grey_range: Tuple[int, int]
    num_particles: int
    attraction_radius: int
    attraction_strength: float
    cell_size: int

@dataclass
class TomographyEffects:
    blur_kernel: int
    noise_scale: float
    noise_correlation_length: int
    poisson_noise_factor: float
    edge_brightness_factor: float
    edge_width: int
    sharpen_amount: float
    fractal_octaves: int = 6
    fractal_persistence: float = 0.6
    simulate_projection: bool = True
    num_angles: int = 1800
    reconstruction_filter: str = 'shepp-logan'
    sinogram_blur_sigma: float = 1.0


def dataclass_from_dict(cls: type[DataclassT], data: ConfigDict) -> DataclassT:
    """Create a dataclass instance from a dictionary, ignoring extra keys."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

class ParticleGrid:
    """Spatial grid for efficient nearby particle lookups."""
    def __init__(self, size: int, cell_size: int):
        self.cell_size = max(1, cell_size)
        self.grid_width = size // self.cell_size + 1
        self.grid: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {i: {} for i in range(self.grid_width * self.grid_width)}

    def get_index(self, x: int, y: int) -> int:
        """Calculate the grid cell index for a given coordinate."""
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        return int(grid_y * self.grid_width + grid_x)

    def get_nearby_particles(self, x: int, y: int, search_radius: int, size: int) -> List[Tuple[int, int, int, int]]:
        """Retrieve particles within a search radius of a given coordinate."""
        cells_to_check = set()
        radius_in_cells = (search_radius // self.cell_size) + 1

        center_idx = self.get_index(x, y)
        center_grid_x = (center_idx % self.grid_width)
        center_grid_y = (center_idx // self.grid_width)

        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_width:
                    cells_to_check.add(grid_y * self.grid_width + grid_x)

        nearby = []
        for idx in cells_to_check:
            if idx in self.grid:
                 nearby.extend(self.grid[idx].values())
        return nearby

    def add_particle(self, particle_id: int, x: int, y: int, rx: int, ry: int):
        """Add a particle's information to the grid."""
        grid_idx = self.get_index(x, y)
        if grid_idx not in self.grid:
             self.grid[grid_idx] = {}
        self.grid[grid_idx][particle_id] = (x, y, rx, ry)

class ParticlePlacer:
    """Handles the placement logic for particles within the inner circle."""
    def __init__(self, params: ImageParameters):
        self.params = params
        self.grid = ParticleGrid(params.size, params.cell_size)

    def generate_particle_dimensions(self) -> Tuple[int, int]:
        """Generate random radii for an elliptical particle based on a Gamma distribution
           derived from experimental equivalent diameters."""
        gamma_shape_k = 4.5961
        gamma_scale_theta = 3.3044
        generated_diameter = np.random.gamma(shape=gamma_shape_k, scale=gamma_scale_theta, size=1)[0]
        generated_diameter = max(1.0, generated_diameter)
        base_radius = int(round(generated_diameter / 2.0))
        base_radius = max(1, base_radius)
        ecc = uniform(-0.15, 0.15)
        radius_x = max(1, int(round(base_radius * (1 + ecc))))
        radius_y = max(1, int(round(base_radius * (1 - ecc))))

        return radius_x, radius_y


    def try_place_particle(self, inner_radius: int, center_offset: List[int]) -> Optional[Tuple]:
        """Attempt to find a valid position for a new particle."""
        radius_x, radius_y = self.generate_particle_dimensions()
        num_candidates = 10
        candidates = []
        scores = []

        center_x = self.params.size // 2 + center_offset[0]
        center_y = self.params.size // 2 + center_offset[1]

        for _ in range(num_candidates):
            angle = uniform(0, 2 * np.pi)
            max_r = inner_radius - max(radius_x, radius_y)
            if max_r <= 0:
                continue
            r = max_r * np.sqrt(uniform(0, 1))

            x = center_x + int(r * np.cos(angle))
            y = center_y + int(r * np.sin(angle))

            nearby_particles = self.grid.get_nearby_particles(
                x, y, max(self.params.attraction_radius, max(radius_x, radius_y) + 10), self.params.size
            )

            score = self._calculate_score(x, y, nearby_particles)
            valid = self._check_validity(x, y, radius_x, radius_y, nearby_particles)

            if valid:
                candidates.append((x, y))
                scores.append(score)

        if not candidates:
            return None

        chosen_idx = self._choose_position(scores)
        x, y = candidates[chosen_idx]
        angle = uniform(0, 360)

        return (x, y, radius_x, radius_y, angle)

    def _calculate_score(self, x: int, y: int, nearby_particles: List) -> float:
        """Calculate an attraction score based on nearby particles."""
        if not nearby_particles:
            return 1.0

        distances = []
        for particle in nearby_particles:
            ex_x, ex_y = particle[:2]
            dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if dist < self.params.attraction_radius:
                distances.append(dist)

        if not distances:
            return 1.0

        avg_dist = np.mean(distances)
        num_close = len(distances)
        density_factor = min(num_close / 5, 1.0)
        proximity_factor = 1 - (avg_dist / self.params.attraction_radius)
        return 1.0 + (proximity_factor * density_factor * self.params.attraction_strength)

    def _check_validity(self, x: int, y: int, radius_x: int, radius_y: int,
                        nearby_particles: List) -> bool:
        """Check if the proposed particle position overlaps with existing particles."""
        check_radius = max(radius_x, radius_y)
        for particle in nearby_particles:
            ex_x, ex_y, ex_rx, ex_ry = particle
            dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if dist < (check_radius + max(ex_rx, ex_ry) - 5):
                return False
        return True

    def _choose_position(self, scores: List[float]) -> int:
        """Choose a candidate position based on weighted random selection using scores."""
        total_score = sum(scores)
        if total_score == 0 or len(scores) == 0:
            return randint(0, len(scores) - 1) if scores else 0
        probs = [s / total_score for s in scores]
        return np.random.choice(len(scores), p=probs)
class XCTSliceGenerator:
    """Generates the synthetic XCT slice and the multi-label segmentation image."""
    def __init__(self, params: ImageParameters, effects: TomographyEffects):
        self.params = params
        self.effects = effects

    def create_base_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the grayscale image and label map."""
        img = np.full((self.params.size, self.params.size),
                      self.params.base_grey, dtype=np.uint16)

        label_image = np.full((self.params.size, self.params.size),
                              LABEL_OUT_OF_RECONSTRUCTION, dtype=np.uint8)

        center = (self.params.size // 2, self.params.size // 2)

        cv2.circle(img, center, self.params.outer_circle_size // 2,
                   self.params.outer_circle_grey, -1)
        cv2.circle(label_image, center, self.params.outer_circle_size // 2,
                   LABEL_BACKGROUND, -1)

        return img, label_image

    def _apply_pre_projection_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply effects inherent to the sample before projection (e.g., edge brightening)."""
        img_float = image.astype(float)

        if self.effects.edge_brightness_factor > 1.0 and self.effects.edge_width > 0:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            outer_radius = self.params.outer_circle_size // 2

            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

            gradient_mask = np.clip((outer_radius - dist_from_center) / self.effects.edge_width, 0, 1)
            gradient_mask = 1 - gradient_mask
            edge_gradient = gradient_mask * (dist_from_center <= outer_radius) * \
                            (dist_from_center >= (outer_radius - self.effects.edge_width))

            brightness_increase = img_float * (self.effects.edge_brightness_factor - 1) * edge_gradient
            img_float = img_float + brightness_increase

        return np.clip(img_float, 0, 65535)


    def _apply_sinogram_effects(self, sinogram: np.ndarray) -> np.ndarray:
        """Apply acquisition effects in sinogram space (blur, noise)."""
        sino_float = sinogram.astype(float)

        if self.effects.sinogram_blur_sigma > 0:
            sino_float = gaussian_filter1d(sino_float,
                                           sigma=self.effects.sinogram_blur_sigma,
                                           axis=0,
                                           mode='nearest')

        if self.effects.poisson_noise_factor > 0:
            epsilon = 1e-9
            effective_intensity = np.maximum(sino_float / (self.effects.poisson_noise_factor + epsilon), 0)
            noisy_counts = np.random.poisson(effective_intensity)
            sino_float = noisy_counts * (self.effects.poisson_noise_factor + epsilon)

        sino_float = np.maximum(sino_float, 0)

        return sino_float

    def _apply_post_reconstruction_effects(self, image: np.ndarray, recon_mask: np.ndarray) -> np.ndarray:
        """Apply effects after reconstruction (residual noise, blur, sharpening)."""
        img_float = image.astype(float)

        if self.effects.noise_scale > 0:
            fractal_noise = generate_fractal_noise(
                image.shape,
                octaves=self.effects.fractal_octaves,
                persistence=self.effects.fractal_persistence,
                scale=self.effects.noise_scale * 0.7
            )
            variation_mask = cv2.GaussianBlur(
                np.random.normal(1, 0.4, image.shape),
                (0, 0),
                max(1, self.effects.noise_correlation_length * 5)
            )
            variation_mask = np.clip(variation_mask, 0.3, 1.7)
            correlated_noise = fractal_noise * variation_mask
            img_float[recon_mask] += correlated_noise[recon_mask]

        if self.effects.blur_kernel > 1:
             kernel_size = self.effects.blur_kernel
             if kernel_size % 2 == 0:
                 kernel_size += 1

             masked_img_for_blur = img_float * recon_mask
             blurred_masked = cv2.blur(masked_img_for_blur, (kernel_size, kernel_size))

             mask_blurred = cv2.blur(recon_mask.astype(float), (kernel_size, kernel_size))
             safe_mask_blurred = np.where(mask_blurred > 1e-6, mask_blurred, 1.0)
             normalized_blurred = blurred_masked / safe_mask_blurred

             img_float[recon_mask] = normalized_blurred[recon_mask]

        if self.effects.sharpen_amount > 0:
            sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * self.effects.sharpen_amount \
                           + np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) * (1 - self.effects.sharpen_amount)

            img_sharpened = cv2.filter2D(img_float, -1, sharpen_filter)

            img_float[recon_mask] = img_sharpened[recon_mask]
        return np.clip(img_float, 0, 65535).astype(np.uint16)


    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the synthetic slice and its corresponding label image."""
        img_ideal, label_image = self.create_base_images()

        inner_circle_center_offset = [np.random.randint(-50, 50),
                                      np.random.randint(-50, 50)]
        inner_radius = self.params.inner_circle_size // 2
        center_inner = (self.params.size // 2 + inner_circle_center_offset[0],
                        self.params.size // 2 + inner_circle_center_offset[1])

        cv2.circle(img_ideal, center_inner, inner_radius,
                   self.params.inner_circle_grey, -1)
        cv2.circle(label_image, center_inner, inner_radius,
                   LABEL_ELECTROLYTE, -1)

        placer = ParticlePlacer(self.params)
        particles_placed = 0
        attempts = 0
        max_attempts = self.params.num_particles * 100

        print("\nPlacing particles...")
        with tqdm(total=self.params.num_particles, desc="Particles", position=0, leave=True) as pbar:
            while particles_placed < self.params.num_particles and attempts < max_attempts:
                particle_data = placer.try_place_particle(inner_radius, inner_circle_center_offset)

                if particle_data:
                    x, y, rx, ry, angle = particle_data
                    grey = randint(*self.params.particle_grey_range)

                    cv2.ellipse(img_ideal, (x, y), (rx, ry), angle, 0, 360, grey, -1)
                    cv2.ellipse(label_image, (x, y), (rx, ry), angle, 0, 360, LABEL_CATHODE, -1)

                    placer.grid.add_particle(particles_placed, x, y, rx, ry)
                    particles_placed += 1
                    pbar.update(1)

                attempts += 1

        print(f"\nPlacement complete:")
        print(f"- Target particles: {self.params.num_particles}")
        print(f"- Particles placed: {particles_placed}")
        print(f"- Attempts made: {attempts}")
        if attempts > 0:
             print(f"- Success rate: {particles_placed / attempts * 100:.1f}%")
        else:
             print("- Success rate: N/A (no attempts made)")

        final_img = None
        recon_mask = label_image > LABEL_OUT_OF_RECONSTRUCTION

        if self.effects.simulate_projection:
            print("\nSimulating Tomography Acquisition/Reconstruction...")

            print(" - Applying pre-projection effects...")
            img_prepped = self._apply_pre_projection_effects(img_ideal)

            print(f" - Creating sinogram ({self.effects.num_angles} angles)...")
            angles = np.linspace(0., 180., self.effects.num_angles, endpoint=False)
            sinogram = radon(img_prepped, theta=angles, circle=True)

            print(" - Applying sinogram noise and blur...")
            sinogram_noisy = self._apply_sinogram_effects(sinogram)

            print(f" - Reconstructing image (filter: {self.effects.reconstruction_filter})...")
            img_reconstructed_float = iradon(sinogram_noisy,
                                             theta=angles,
                                             filter_name=self.effects.reconstruction_filter,
                                             interpolation='linear',
                                             circle=True)

            if img_reconstructed_float.shape != img_ideal.shape:
                 print(f"Warning: Reconstructed size {img_reconstructed_float.shape} != ideal size {img_ideal.shape}. Cropping/Padding.")
                 target_shape = np.array(img_ideal.shape)
                 current_shape = np.array(img_reconstructed_float.shape)
                 diff = target_shape - current_shape
                 pad_before = diff // 2
                 pad_after = diff - pad_before

                 pads = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]))

                 if np.any(diff > 0):
                     pads_non_neg = tuple((max(0, p[0]), max(0, p[1])) for p in pads)
                     img_reconstructed_float = np.pad(img_reconstructed_float, pads_non_neg,
                                                      mode='constant', constant_values=np.mean(img_reconstructed_float))
                 elif np.any(diff < 0):
                     crops = tuple((max(0, -p[0]), max(0, s + p[1])) for p, s in zip(pads, current_shape))
                     img_reconstructed_float = img_reconstructed_float[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]

                 if img_reconstructed_float.shape != img_ideal.shape:
                    print(f"Error: Size mismatch after crop/pad ({img_reconstructed_float.shape}). Resizing forcefully.")
                    img_reconstructed_float = cv2.resize(img_reconstructed_float, (img_ideal.shape[1], img_ideal.shape[0]), interpolation=cv2.INTER_LINEAR)

            print(" - Applying post-reconstruction effects...")
            final_img = self._apply_post_reconstruction_effects(img_reconstructed_float, recon_mask)

        else:
             print("\nSkipping projection simulation. Applying post-reconstruction effects directly to ideal image...")
             final_img = self._apply_post_reconstruction_effects(img_ideal.astype(float), recon_mask)

        return final_img, label_image
def generate_fractal_noise(shape: Tuple[int, int], octaves: int = 4,
                            persistence: float = 0.5, scale: float = 1.0) -> np.ndarray:
    """
    Generate fractal noise using multiple octaves of Perlin-like noise.
    Creates more natural-looking, textured noise patterns.
    """
    noise = np.zeros(shape, dtype=float)
    frequency = 1
    amplitude = 1.0
    max_amplitude = 0 # Track max possible amplitude for normalization

    for octave in range(octaves):
        max_amplitude += amplitude
        # Generate base noise (Gaussian blur on random noise creates smoother base)
        base_noise = np.random.normal(0, 1, shape)
        # Adjust sigma based on frequency to control feature size
        # Ensure sigma is reasonable and scales with image size/frequency
        sigma_base = max(shape) / 8 # Base sigma related to image size
        sigma = max(1.0, sigma_base / frequency) # Decrease sigma for higher frequencies
        noise_layer = cv2.GaussianBlur(base_noise, (0,0), sigmaX=sigma, sigmaY=sigma)

        # Add the noise layer, scaled by amplitude
        noise += amplitude * noise_layer

        # Update amplitude and frequency for the next octave
        amplitude *= persistence
        frequency *= 2 # Double frequency for finer details

    # Normalize the combined noise approximately to the [-1, 1] range, then scale
    if max_amplitude > 1e-6: # Avoid division by zero
         # Normalize by max possible amplitude and then scale
         noise = (noise / max_amplitude) * scale
         # Alternative: Normalize by actual std dev if preferred
         # noise_std = np.std(noise)
         # if noise_std > 1e-6:
         #    noise = (noise / noise_std) * scale
         # else:
         #    noise = np.zeros(shape)
    else:
        noise = np.zeros(shape) # Return zero noise if max_amplitude is negligible

    return noise

def plot_results(synthetic_slice: np.ndarray, label_image: np.ndarray):
    """Plot the generated slice and label image."""
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    ax1 = axes[0]
    vmin = np.percentile(synthetic_slice[synthetic_slice > 0], 1) if np.any(synthetic_slice > 0) else 0
    vmax = np.percentile(synthetic_slice, 99) if np.any(synthetic_slice > 0) else 65535
    im1 = ax1.imshow(synthetic_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('Synthetic XCT Slice (Greyscale)')
    ax1.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity (uint16)')

    ax2 = axes[1]
    num_labels = 4
    cmap = plt.get_cmap('viridis', num_labels)

    bounds = np.arange(num_labels + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    im2 = ax2.imshow(label_image, cmap=cmap, norm=norm, interpolation='nearest')
    ax2.set_title('Segmentation Label Image')
    ax2.axis('off')

    label_names = {
        LABEL_OUT_OF_RECONSTRUCTION: 'Out of Recon (0)',
        LABEL_BACKGROUND: 'Background (1)',
        LABEL_ELECTROLYTE: 'Electrolyte (2)',
        LABEL_CATHODE: 'Cathode (3)'
    }
    patches = [mpatches.Patch(color=cmap(norm(label_value)), label=name)
               for label_value, name in sorted(label_names.items())]

    ax2.legend(handles=patches,
               loc='lower left',
               bbox_to_anchor=(0.01, 0.01),
               borderaxespad=0.,
               fontsize='small',
               title="Phase Legend",
               title_fontsize='medium')

    plt.tight_layout()
    plt.show()

def randomize_config(base_config: ConfigDict) -> ConfigDict:
    """Create a randomized configuration derived from the base config."""
    config = deepcopy(base_config)

    img_params = config['image']
    img_params['inner_circle_grey'] += randint(-3000, 3000)
    img_params['outer_circle_grey'] += randint(-3000, 3000)
    particle_min_delta = randint(-1000, 1000)
    particle_max_delta = randint(-5000, 5000)
    new_min = img_params['particle_grey_range'][0] + particle_min_delta
    new_max = img_params['particle_grey_range'][1] + particle_max_delta
    img_params['particle_grey_range'] = (max(1, new_min), max(new_min + 1, new_max))

    img_params['num_particles'] += randint(-500, 1000)
    img_params['attraction_radius'] += randint(-10, 10)
    img_params['attraction_strength'] *= uniform(0.8, 1.2)
    img_params['cell_size'] += randint(-30, 30)

    img_params['inner_circle_grey'] = max(1, min(65535, img_params['inner_circle_grey']))
    img_params['outer_circle_grey'] = max(1, min(65535, img_params['outer_circle_grey']))
    img_params['num_particles'] = max(100, img_params['num_particles'])
    img_params['attraction_radius'] = max(5, img_params['attraction_radius'])
    img_params['attraction_strength'] = max(0, img_params['attraction_strength'])
    img_params['cell_size'] = max(20, min(200, img_params['cell_size']))

    effects_params = config['effects']
    blur_delta = randint(-4, 4) // 2 * 2
    effects_params['blur_kernel'] += blur_delta
    effects_params['noise_scale'] += uniform(-1000, 1000)
    effects_params['noise_correlation_length'] += randint(-1, 1)
    effects_params['poisson_noise_factor'] += uniform(-0.1, 0.1)
    effects_params['edge_brightness_factor'] += uniform(-0.1, 0.1)
    effects_params['edge_width'] += randint(-20, 20)
    effects_params['sharpen_amount'] += uniform(-0.2, 0.2)
    effects_params['fractal_octaves'] += randint(-1, 1)
    effects_params['fractal_persistence'] += uniform(-0.1, 0.1)

    if effects_params.get('simulate_projection', False):
        effects_params['num_angles'] += randint(-300, 300)
        effects_params['sinogram_blur_sigma'] += uniform(-0.5, 0.5)

    effects_params['blur_kernel'] = max(3, min(15, effects_params['blur_kernel']))
    if effects_params['blur_kernel'] % 2 == 0:
        effects_params['blur_kernel'] += 1

    effects_params['noise_scale'] = max(100.0, effects_params['noise_scale'])
    effects_params['noise_correlation_length'] = max(1, effects_params['noise_correlation_length'])
    effects_params['poisson_noise_factor'] = max(0.01, min(5.0, effects_params['poisson_noise_factor']))
    effects_params['edge_brightness_factor'] = max(1.0, min(1.5, effects_params['edge_brightness_factor']))
    effects_params['edge_width'] = max(10, min(150, effects_params['edge_width']))
    effects_params['sharpen_amount'] = max(0.0, min(1.0, effects_params['sharpen_amount']))
    effects_params['fractal_octaves'] = max(1, min(8, effects_params['fractal_octaves']))
    effects_params['fractal_persistence'] = max(0.1, min(0.9, effects_params['fractal_persistence']))

    if effects_params.get('simulate_projection', False):
        effects_params['num_angles'] = max(180, min(3600, effects_params['num_angles']))
        effects_params['sinogram_blur_sigma'] = max(0.1, min(5.0, effects_params['sinogram_blur_sigma']))
        valid_filters = ['shepp-logan', 'ram-lak', 'cosine', 'hamming', 'hann']
        if effects_params['reconstruction_filter'] not in valid_filters:
            effects_params['reconstruction_filter'] = 'shepp-logan'

    return config

if __name__ == "__main__":
    print("Generating synthetic XCT slice with multi-label mask...")

    config = {
        "generation": {
            "output_dir": Path("./synthetic_data_simulated_recon"),
            "num_slices": 100,
            "crop_size": 2000,
            "randomize": True,
            "plot_single_slice": True
        },
        "image": {
            "size": 2560,
            "outer_circle_size": 2560,
            "inner_circle_size": 1860,
            "base_grey": 0,
            "inner_circle_grey": 40000,
            "outer_circle_grey": 50000,
            "particle_grey_range": (2560, 17920),
            "num_particles": 4000,
            "attraction_radius": 30,
            "attraction_strength": 1.0e5,
            "cell_size": 100
        },
        "effects": {
            "blur_kernel": 1,
            "noise_scale": 20000.0,
            "noise_correlation_length": 2,
            "poisson_noise_factor": 20,
            "edge_brightness_factor": 1.2,
            "edge_width": 100,
            "sharpen_amount": 0.3,
            "fractal_octaves": 6,
            "fractal_persistence": 0.8,
            "simulate_projection": True,
            "num_angles": 800,
            "reconstruction_filter": 'shepp-logan',
            "sinogram_blur_sigma": 5,
        }
    }

    output_dir = Path(config['generation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_slices_to_generate = config['generation']['num_slices']

    for i in range(num_slices_to_generate):
        print(f"\n--- Generating Slice {i+1}/{num_slices_to_generate} ---")

        if config['generation']['randomize']:
            slice_config = randomize_config(config)
            print("\nUsing Randomized Parameters:")
        else:
            slice_config = deepcopy(config)
            print("\nUsing Base Parameters:")

        image_params = dataclass_from_dict(ImageParameters, slice_config['image'])
        tomography_effects = dataclass_from_dict(TomographyEffects, slice_config['effects'])

        generator = XCTSliceGenerator(image_params, tomography_effects)
        synthetic_slice, label_image = generator.generate()

        crop_size = slice_config['generation']['crop_size']
        image_size = slice_config['image']['size']

        if crop_size > 0 and crop_size < image_size:
            print(f"\nCropping images to center {crop_size}x{crop_size} pixels...")
            start_x = (image_size - crop_size) // 2
            start_y = (image_size - crop_size) // 2
            end_x = start_x + crop_size
            end_y = start_y + crop_size
            synthetic_slice_cropped = synthetic_slice[start_y:end_y, start_x:end_x]
            label_image_cropped = label_image[start_y:end_y, start_x:end_x]
        elif crop_size > 0 and crop_size >= image_size:
             print(f"\nSkipping cropping as crop_size ({crop_size}) >= image size ({image_size}).")
             synthetic_slice_cropped = synthetic_slice
             label_image_cropped = label_image
        else:
             print("\nSkipping cropping as crop_size is not set or invalid.")
             synthetic_slice_cropped = synthetic_slice
             label_image_cropped = label_image

        if num_slices_to_generate == 1 and config['generation']['plot_single_slice']:
             print("\nPlotting results...")
             plot_results(synthetic_slice_cropped, label_image_cropped)

        slice_filename = output_dir / f"synthetic_slice_{i:03d}.tif"
        label_filename = output_dir / f"label_image_{i:03d}.tif"

        print(f"\nSaving greyscale slice to: {slice_filename}")
        imwrite(slice_filename, synthetic_slice_cropped, imagej=True, metadata={'axes': 'YX'})

        print(f"Saving label image to: {label_filename}")
        imwrite(label_filename, label_image_cropped.astype(np.uint8), imagej=True, metadata={'axes': 'YX'})

    print(f"\nFinished generating {num_slices_to_generate} slice(s) in '{output_dir}'.")