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

import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import patches for legend
from matplotlib.colors import ListedColormap, BoundaryNorm
from random import uniform, randint
from copy import deepcopy
import os
from tifffile import imwrite
from pathlib import Path
import pprint # For printing the config dictionary nicely

# --- Tomography Simulation Imports ---
from skimage.transform import radon, iradon # For projection/reconstruction
from scipy.ndimage import gaussian_filter1d # For 1D blur in sinogram

# --- Constants for Labels ---
LABEL_OUT_OF_RECONSTRUCTION = 0
LABEL_BACKGROUND = 0
LABEL_ELECTROLYTE = 1
LABEL_CATHODE = 2
# --------------------------

# --- Dataclasses for Parameter Structure (Type Hinting & Organization) ---
# These dataclasses now primarily serve for structure and type hints within classes.
# They are initialized from the main configuration dictionary.

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
    # Existing parameters (roles potentially adjusted)
    blur_kernel: int # Now potentially post-reconstruction blur
    noise_scale: float # Correlated noise scale (post-recon)
    noise_correlation_length: int # Correlated noise length (post-recon)
    poisson_noise_factor: float # Now primarily for sinogram Poisson noise
    edge_brightness_factor: float # Applied pre-projection
    edge_width: int             # Applied pre-projection
    sharpen_amount: float         # Applied post-reconstruction
    fractal_octaves: int = 6      # Correlated noise param
    fractal_persistence: float = 0.6 # Correlated noise param

    # New parameters for projection/reconstruction simulation
    simulate_projection: bool = True  # Flag to enable/disable this feature
    num_angles: int = 1800            # Number of projection angles
    reconstruction_filter: str = 'shepp-logan' # Filter for iradon
    sinogram_blur_sigma: float = 1.0  # Sigma for 1D Gaussian blur on sinogram rows
    # sinogram_poisson_factor: float = 1.0 # Optional: Separate factor if needed


# --- Helper Function to Create Dataclasses from Dict ---
def dataclass_from_dict(cls, data: Dict[str, Any]):
    """Creates a dataclass instance from a dictionary, ignoring extra keys."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

# --- Particle Grid (Spatial Acceleration) ---
class ParticleGrid:
    """Spatial grid for efficient nearby particle lookups."""
    def __init__(self, size: int, cell_size: int):
        self.cell_size = max(1, cell_size) # Ensure cell_size is at least 1
        self.grid_width = size // self.cell_size + 1
        # Initialize grid as a dictionary of dictionaries for sparse storage
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

        # Determine the grid cells to check based on the search radius
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                # Ensure the cell is within the grid boundaries
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_width:
                    cells_to_check.add(grid_y * self.grid_width + grid_x)

        nearby = []
        # Collect particles from the identified cells
        for idx in cells_to_check:
            if idx in self.grid: # Check if index exists (might not if grid is sparse)
                 nearby.extend(self.grid[idx].values())
        return nearby

    def add_particle(self, particle_id: int, x: int, y: int, rx: int, ry: int):
        """Add a particle's information to the grid."""
        grid_idx = self.get_index(x, y)
        if grid_idx not in self.grid:
             self.grid[grid_idx] = {} # Initialize cell if it doesn't exist
        self.grid[grid_idx][particle_id] = (x, y, rx, ry)

# --- Particle Placement Logic ---
class ParticlePlacer:
    """Handles the placement logic for particles within the inner circle."""
    def __init__(self, params: ImageParameters):
        self.params = params
        # Initialize grid using parameters from the ImageParameters dataclass
        self.grid = ParticleGrid(params.size, params.cell_size)

    def generate_particle_dimensions(self) -> Tuple[int, int]:
        """Generate random radii for an elliptical particle based on a Gamma distribution
           derived from experimental equivalent diameters."""

        # --- Parameters from your Gamma fit ---
        gamma_shape_k = 4.5961
        gamma_scale_theta = 3.3044
        # ---------------------------------------

        # 1. Generate an equivalent diameter from the fitted Gamma distribution
        generated_diameter = np.random.gamma(shape=gamma_shape_k, scale=gamma_scale_theta, size=1)[0]

        # Ensure diameter is not unrealistically small (e.g., less than 1 pixel)
        generated_diameter = max(1.0, generated_diameter)

        # 2. Calculate a base radius from the diameter
        base_radius = int(round(generated_diameter / 2.0))

        # Ensure base_radius is at least 1 pixel
        base_radius = max(1, base_radius)

        # 3. Introduce slight eccentricity
        ecc = uniform(-0.15, 0.15)
        radius_x = max(1, int(round(base_radius * (1 + ecc)))) # Use round before int
        radius_y = max(1, int(round(base_radius * (1 - ecc)))) # Ensure radius is at least 1

        return radius_x, radius_y


    def try_place_particle(self, inner_radius: int, center_offset: List[int]) -> Optional[Tuple]:
        """Attempt to find a valid position for a new particle."""
        radius_x, radius_y = self.generate_particle_dimensions()
        num_candidates = 10 # Number of random positions to try
        candidates = []
        scores = []

        center_x = self.params.size // 2 + center_offset[0]
        center_y = self.params.size // 2 + center_offset[1]

        for _ in range(num_candidates):
            # Generate random position within the inner circle
            angle = uniform(0, 2 * np.pi)
            max_r = inner_radius - max(radius_x, radius_y) # Max distance from center
            if max_r <= 0: continue # Skip if particle is too large for the circle
            r = max_r * np.sqrt(uniform(0, 1)) # Uniform area sampling

            x = center_x + int(r * np.cos(angle))
            y = center_y + int(r * np.sin(angle))

            # Check for collisions and calculate attraction score
            nearby_particles = self.grid.get_nearby_particles(
                x, y, max(self.params.attraction_radius, max(radius_x, radius_y) + 10), self.params.size
            )

            score = self._calculate_score(x, y, nearby_particles)
            valid = self._check_validity(x, y, radius_x, radius_y, nearby_particles)

            if valid:
                candidates.append((x, y))
                scores.append(score)

        if not candidates:
            return None # No valid position found

        # Choose the best candidate based on scores
        chosen_idx = self._choose_position(scores)
        x, y = candidates[chosen_idx]
        angle = uniform(0, 360) # Random orientation

        return (x, y, radius_x, radius_y, angle)

    def _calculate_score(self, x: int, y: int, nearby_particles: List) -> float:
        """Calculate an attraction score based on nearby particles."""
        if not nearby_particles:
            return 1.0 # Base score if no neighbors

        distances = []
        for particle in nearby_particles:
            ex_x, ex_y = particle[:2]
            dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if dist < self.params.attraction_radius:
                distances.append(dist)

        if not distances:
            return 1.0 # Base score if no neighbors within attraction radius

        # Score higher if closer to more particles (promotes clustering)
        avg_dist = np.mean(distances)
        num_close = len(distances)
        density_factor = min(num_close / 5, 1.0) # Cap density influence
        proximity_factor = 1 - (avg_dist / self.params.attraction_radius)
        return 1.0 + (proximity_factor * density_factor * self.params.attraction_strength)

    def _check_validity(self, x: int, y: int, radius_x: int, radius_y: int,
                        nearby_particles: List) -> bool:
        """Check if the proposed particle position overlaps with existing particles."""
        check_radius = max(radius_x, radius_y)
        for particle in nearby_particles:
            ex_x, ex_y, ex_rx, ex_ry = particle
            dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            # Check for overlap, allowing a small tolerance (-5 pixels)
            if dist < (check_radius + max(ex_rx, ex_ry) - 5):
                return False # Overlap detected
        return True # No overlap

    def _choose_position(self, scores: List[float]) -> int:
        """Choose a candidate position based on weighted random selection using scores."""
        total_score = sum(scores)
        if total_score == 0 or len(scores) == 0:
             # Fallback to random choice if no scores or all scores are zero
            return randint(0, len(scores) - 1) if scores else 0
        # Normalize scores to probabilities
        probs = [s / total_score for s in scores]
        # Choose index based on probability distribution
        return np.random.choice(len(scores), p=probs)

# --- Main Generator Class ---
class XCTSliceGenerator:
    """Generates the synthetic XCT slice and the multi-label segmentation image."""
    def __init__(self, params: ImageParameters, effects: TomographyEffects):
        self.params = params
        self.effects = effects

    def create_base_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes the greyscale image and the multi-label image."""
        # Greyscale image (uint16 for high dynamic range)
        img = np.full((self.params.size, self.params.size),
                      self.params.base_grey, dtype=np.uint16)

        # Multi-label image (uint8 for labels 0-3)
        # Start with everything outside the reconstruction volume (label 0)
        label_image = np.full((self.params.size, self.params.size),
                              LABEL_OUT_OF_RECONSTRUCTION, dtype=np.uint8)

        center = (self.params.size // 2, self.params.size // 2)

        # Draw outer circle (reconstruction volume boundary)
        # This area is initially the 'background' ring
        cv2.circle(img, center, self.params.outer_circle_size // 2,
                   self.params.outer_circle_grey, -1)
        cv2.circle(label_image, center, self.params.outer_circle_size // 2,
                   LABEL_BACKGROUND, -1)

        return img, label_image

    def _apply_pre_projection_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply effects inherent to the sample before projection (e.g., edge brightening)."""
        img_float = image.astype(float) # Work with float for calculations

        # --- Edge Brightening Effect ---
        if self.effects.edge_brightness_factor > 1.0 and self.effects.edge_width > 0:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            outer_radius = self.params.outer_circle_size // 2

            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

            # Create a gradient mask that peaks at the edge and falls off inwards
            gradient_mask = np.clip((outer_radius - dist_from_center) / self.effects.edge_width, 0, 1)
            gradient_mask = 1 - gradient_mask # Invert: 1 at edge, 0 further in
            # Apply only within the reconstruction volume and close to the edge
            edge_gradient = gradient_mask * (dist_from_center <= outer_radius) * \
                            (dist_from_center >= (outer_radius - self.effects.edge_width))

            # Calculate brightness increase based on the gradient and factor
            brightness_increase = img_float * (self.effects.edge_brightness_factor - 1) * edge_gradient
            img_float = img_float + brightness_increase

        # Clip to ensure values stay within reasonable bounds (though uint16 allows large range)
        # Keep as float for radon transform
        return np.clip(img_float, 0, 65535)


    def _apply_sinogram_effects(self, sinogram: np.ndarray) -> np.ndarray:
        """Apply acquisition effects in sinogram space (blur, noise)."""
        sino_float = sinogram.astype(float)

        # --- 1. Detector Blur (1D Gaussian along detector axis) ---
        # skimage radon output shape: (n_detector_pixels, n_angles)
        # We blur along axis 0.
        if self.effects.sinogram_blur_sigma > 0:
            sino_float = gaussian_filter1d(sino_float,
                                           sigma=self.effects.sinogram_blur_sigma,
                                           axis=0, # Apply along detector pixel dimension
                                           mode='nearest') # Handle edges

        # --- 2. Poisson Noise (Shot Noise) ---
        # This simulates noise due to photon counting statistics.
        if self.effects.poisson_noise_factor > 0:
            # Intensity-dependent noise applied to the blurred sinogram.
            # The factor scales the intensity before Poisson sampling.
            # A higher factor simulates higher counts (relatively less noise).
            epsilon = 1e-9 # Avoid division by zero or log(0)
            # Scale intensities *down* to simulate lower counts, apply Poisson, then scale back *up*.
            effective_intensity = np.maximum(sino_float / (self.effects.poisson_noise_factor + epsilon), 0)
            noisy_counts = np.random.poisson(effective_intensity)
            # Scale back to original intensity range
            sino_float = noisy_counts * (self.effects.poisson_noise_factor + epsilon)

        # --- 3. Optional: Add simple Gaussian noise (readout etc.) ---
        # Example - could add another config parameter for this
        # readout_noise_std = 50.0 # Example value
        # if readout_noise_std > 0:
        #    gaussian_noise = np.random.normal(0, readout_noise_std, sino_float.shape)
        #    sino_float += gaussian_noise

        # Ensure non-negative, although Poisson should handle this mostly
        sino_float = np.maximum(sino_float, 0)

        return sino_float

    def _apply_post_reconstruction_effects(self, image: np.ndarray, recon_mask: np.ndarray) -> np.ndarray:
        """Apply effects after reconstruction (residual noise, blur, sharpening)."""
        img_float = image.astype(float)

        # --- 1. Correlated Noise (e.g., reconstruction artifacts, residual texture) ---
        if self.effects.noise_scale > 0:
            # Generate noise only once if needed multiple times
            # Base fractal noise for texture
            fractal_noise = generate_fractal_noise(
                image.shape,
                octaves=self.effects.fractal_octaves,
                persistence=self.effects.fractal_persistence,
                scale=self.effects.noise_scale * 0.7 # Scale for fractal part
            )
            # Spatially varying intensity mask (smooth random variations)
            variation_mask = cv2.GaussianBlur(
                np.random.normal(1, 0.4, image.shape), # Mean 1, std dev 0.4
                (0, 0), # Kernel size derived from sigma
                max(1, self.effects.noise_correlation_length * 5) # Larger sigma for smoother variation, ensure > 0
            )
            variation_mask = np.clip(variation_mask, 0.3, 1.7) # Limit variation range
            correlated_noise = fractal_noise * variation_mask
            # Apply noise only within the reconstruction mask
            img_float[recon_mask] += correlated_noise[recon_mask]


        # --- 2. Optional Post-Reconstruction Blur ---
        # Simulates final detector PSF or slight image smoothing after reconstruction
        if self.effects.blur_kernel > 1:
             kernel_size = self.effects.blur_kernel
             if kernel_size % 2 == 0: kernel_size += 1 # Ensure odd

             # Apply blur carefully using the mask
             # Create a version where outside is 0 for blurring
             masked_img_for_blur = img_float * recon_mask
             blurred_masked = cv2.blur(masked_img_for_blur, (kernel_size, kernel_size))

             # Normalize the blur effect by blurring the mask itself
             # This prevents darkening near the mask edges
             mask_blurred = cv2.blur(recon_mask.astype(float), (kernel_size, kernel_size))
             # Avoid division by zero where mask_blurred is close to zero
             safe_mask_blurred = np.where(mask_blurred > 1e-6, mask_blurred, 1.0)
             normalized_blurred = blurred_masked / safe_mask_blurred

             # Update the image only within the mask region
             img_float[recon_mask] = normalized_blurred[recon_mask]


        # --- 3. Sharpening ---
        if self.effects.sharpen_amount > 0:
            # Simple sharpening filter applied to the potentially blurred/noisy image
            sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * self.effects.sharpen_amount \
                           + np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) * (1 - self.effects.sharpen_amount) # Blend

            # Apply sharpening - Use filter2D on the float image
            # Applying within the mask is tricky with filter2D due to kernel boundaries.
            # Simplest is to filter the whole image and then apply the mask,
            # accepting minor artifacts at the boundary of the mask.
            img_sharpened = cv2.filter2D(img_float, -1, sharpen_filter) # Using -1 for same depth

            # Apply the sharpened result only within the mask
            img_float[recon_mask] = img_sharpened[recon_mask]


        # --- Final Clipping and Conversion ---
        return np.clip(img_float, 0, 65535).astype(np.uint16)


    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the synthetic slice and its corresponding label image."""
        img_ideal, label_image = self.create_base_images()

        # --- Draw Inner Circle (Electrolyte) ---
        # Introduce random offset for the inner circle center
        inner_circle_center_offset = [np.random.randint(-50, 50),
                                      np.random.randint(-50, 50)]
        inner_radius = self.params.inner_circle_size // 2
        center_inner = (self.params.size // 2 + inner_circle_center_offset[0],
                        self.params.size // 2 + inner_circle_center_offset[1])

        # Draw on greyscale image (ideal version)
        cv2.circle(img_ideal, center_inner, inner_radius,
                   self.params.inner_circle_grey, -1)
        # Draw on label image
        cv2.circle(label_image, center_inner, inner_radius,
                   LABEL_ELECTROLYTE, -1)

        # --- Place Particles (Cathode) ---
        # Pass the ImageParameters dataclass to the placer
        placer = ParticlePlacer(self.params)
        particles_placed = 0
        attempts = 0
        max_attempts = self.params.num_particles * 100 # Limit attempts

        print("\nPlacing particles...")
        with tqdm(total=self.params.num_particles, desc="Particles", position=0, leave=True) as pbar:
            while particles_placed < self.params.num_particles and attempts < max_attempts:
                # Try to place a particle within the inner circle
                particle_data = placer.try_place_particle(inner_radius, inner_circle_center_offset)

                if particle_data:
                    x, y, rx, ry, angle = particle_data
                    # Assign random grey value within the specified range
                    grey = randint(*self.params.particle_grey_range)

                    # Draw particle on greyscale image (ideal version)
                    cv2.ellipse(img_ideal, (x, y), (rx, ry), angle, 0, 360, grey, -1)
                    # Draw particle on label image with cathode label
                    cv2.ellipse(label_image, (x, y), (rx, ry), angle, 0, 360, LABEL_CATHODE, -1)

                    # Add particle to the spatial grid for collision checking
                    placer.grid.add_particle(particles_placed, x, y, rx, ry)
                    particles_placed += 1
                    pbar.update(1)

                attempts += 1

        # Print placement statistics
        print(f"\nPlacement complete:")
        print(f"- Target particles: {self.params.num_particles}")
        print(f"- Particles placed: {particles_placed}")
        print(f"- Attempts made: {attempts}")
        if attempts > 0:
             print(f"- Success rate: {particles_placed / attempts * 100:.1f}%")
        else:
             print("- Success rate: N/A (no attempts made)")


        # --- Apply Tomography Simulation or Direct Effects ---
        final_img = None # Initialize
        # Create the reconstruction mask (where labels > 0) used in both paths
        recon_mask = label_image > LABEL_OUT_OF_RECONSTRUCTION

        if self.effects.simulate_projection:
            print("\nSimulating Tomography Acquisition/Reconstruction...")

            # 1. Apply Pre-Projection Effects (e.g., edge brightening)
            print(" - Applying pre-projection effects...")
            # Pass the ideal image, returns float version with effects
            img_prepped = self._apply_pre_projection_effects(img_ideal)

            # 2. Forward Projection (Radon Transform)
            print(f" - Creating sinogram ({self.effects.num_angles} angles)...")
            angles = np.linspace(0., 180., self.effects.num_angles, endpoint=False)
            # radon function expects float input
            sinogram = radon(img_prepped, theta=angles, circle=True) # img_prepped is already float

            # 3. Apply Sinogram Space Effects (Noise, Blur)
            print(" - Applying sinogram noise and blur...")
            sinogram_noisy = self._apply_sinogram_effects(sinogram)

            # 4. Reconstruction (Inverse Radon Transform)
            print(f" - Reconstructing image (filter: {self.effects.reconstruction_filter})...")
            # iradon returns float64 usually
            img_reconstructed_float = iradon(sinogram_noisy,
                                             theta=angles,
                                             filter_name=self.effects.reconstruction_filter,
                                             interpolation='linear', # 'cubic' is smoother but slower
                                             circle=True) # Helps match output size

            # Ensure reconstruction size matches original, pad/crop if necessary
            # (This is less common with circle=True but good to have)
            if img_reconstructed_float.shape != img_ideal.shape:
                 print(f"Warning: Reconstructed size {img_reconstructed_float.shape} != ideal size {img_ideal.shape}. Cropping/Padding.")
                 # Basic center cropping/padding:
                 target_shape = np.array(img_ideal.shape)
                 current_shape = np.array(img_reconstructed_float.shape)
                 diff = target_shape - current_shape
                 pad_before = diff // 2
                 pad_after = diff - pad_before

                 pads = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]))

                 if np.any(diff > 0): # Need padding
                     pads_non_neg = tuple((max(0, p[0]), max(0, p[1])) for p in pads)
                     img_reconstructed_float = np.pad(img_reconstructed_float, pads_non_neg,
                                                      mode='constant', constant_values=np.mean(img_reconstructed_float)) # Pad with mean or 0
                 elif np.any(diff < 0): # Need cropping
                     crops = tuple((max(0, -p[0]), max(0, s + p[1])) for p, s in zip(pads, current_shape)) # Use negative pads for slice indices
                     img_reconstructed_float = img_reconstructed_float[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]

                 # Final check after padding/cropping
                 if img_reconstructed_float.shape != img_ideal.shape:
                    print(f"Error: Size mismatch after crop/pad ({img_reconstructed_float.shape}). Resizing forcefully.")
                    img_reconstructed_float = cv2.resize(img_reconstructed_float, (img_ideal.shape[1], img_ideal.shape[0]), interpolation=cv2.INTER_LINEAR)


            # 5. Apply Post-Reconstruction Effects (Noise, Blur, Sharpen)
            print(" - Applying post-reconstruction effects...")
            # Pass the float reconstructed image and mask, returns uint16
            final_img = self._apply_post_reconstruction_effects(img_reconstructed_float, recon_mask)

            # Optional: Force area outside mask back to a specific value if needed
            # final_img[~recon_mask] = self.params.base_grey # Example

        else:
             # --- Fallback: Apply effects directly if simulation is off ---
             print("\nSkipping projection simulation. Applying post-reconstruction effects directly to ideal image...")
             # Apply the post-processing steps (noise, blur, sharpen) directly to the ideal image
             # Note: This won't have the reconstruction artifacts but includes other effects.
             # Pass the ideal image as float and the mask, returns uint16
             final_img = self._apply_post_reconstruction_effects(img_ideal.astype(float), recon_mask)


        # Return the final processed image and the ORIGINAL label image
        # final_img should be uint16 from the post_reconstruction function
        return final_img, label_image


# --- Noise Generation Utility ---
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


# --- Plotting Utility ---
def plot_results(synthetic_slice: np.ndarray, label_image: np.ndarray):
    """Plots the generated synthetic slice and its multi-label segmentation with a legend."""
    plt.style.use('default') # Use default matplotlib style
    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # Adjusted figure size

    # --- Plot Synthetic Slice ---
    ax1 = axes[0]
    # Determine reasonable intensity limits for display
    vmin = np.percentile(synthetic_slice[synthetic_slice > 0], 1) if np.any(synthetic_slice > 0) else 0
    vmax = np.percentile(synthetic_slice, 99) if np.any(synthetic_slice > 0) else 65535
    im1 = ax1.imshow(synthetic_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('Synthetic XCT Slice (Greyscale)')
    ax1.axis('off')
    # Add colorbar for greyscale image
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity (uint16)')


    # --- Plot Label Image ---
    ax2 = axes[1]
    # Use 'viridis' or 'plasma' colormap with 4 distinct colors
    num_labels = 4 # 0, 1, 2, 3
    cmap = plt.get_cmap('viridis', num_labels)

    # Define the boundaries for the labels (0, 1, 2, 3)
    bounds = np.arange(num_labels + 1) - 0.5 # Centered boundaries: -0.5, 0.5, 1.5, 2.5, 3.5
    norm = BoundaryNorm(bounds, cmap.N)

    im2 = ax2.imshow(label_image, cmap=cmap, norm=norm, interpolation='nearest')
    ax2.set_title('Segmentation Label Image')
    ax2.axis('off')

    # --- Create Legend instead of Colorbar ---
    label_names = {
        LABEL_OUT_OF_RECONSTRUCTION: 'Out of Recon (0)',
        LABEL_BACKGROUND: 'Background (1)',
        LABEL_ELECTROLYTE: 'Electrolyte (2)',
        LABEL_CATHODE: 'Cathode (3)'
    }
    # Create patches as handles for the legend
    patches = [mpatches.Patch(color=cmap(norm(label_value)), label=name)
               for label_value, name in sorted(label_names.items())] # Sort by label value

    # Add legend to the plot
    ax2.legend(handles=patches,
               loc='lower left',         # Position the legend
               bbox_to_anchor=(0.01, 0.01), # Fine-tune position (relative to axes)
               borderaxespad=0.,        # Padding around legend border
               fontsize='small',         # Adjust font size
               title="Phase Legend",     # Optional legend title
               title_fontsize='medium') # Optional title font size

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()


# --- Parameter Randomization Function ---
def randomize_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new configuration dictionary with randomized values based on the base config.
    """
    config = deepcopy(base_config) # Start with a copy of the base config

    # --- Randomize Image Parameters ---
    img_params = config['image']
    img_params['inner_circle_grey'] += randint(-3000, 3000)
    img_params['outer_circle_grey'] += randint(-3000, 3000)
    # Ensure particle grey range remains valid (min < max)
    particle_min_delta = randint(-1000, 1000)
    particle_max_delta = randint(-5000, 5000)
    new_min = img_params['particle_grey_range'][0] + particle_min_delta
    new_max = img_params['particle_grey_range'][1] + particle_max_delta
    img_params['particle_grey_range'] = (max(1, new_min), max(new_min + 1, new_max)) # Ensure min < max and > 0

    img_params['num_particles'] += randint(-500, 1000)
    img_params['attraction_radius'] += randint(-10, 10)
    img_params['attraction_strength'] *= uniform(0.8, 1.2) # Multiplicative change
    img_params['cell_size'] += randint(-30, 30)

    # Ensure image param values stay within reasonable bounds
    img_params['inner_circle_grey'] = max(1, min(65535, img_params['inner_circle_grey']))
    img_params['outer_circle_grey'] = max(1, min(65535, img_params['outer_circle_grey']))
    img_params['num_particles'] = max(100, img_params['num_particles']) # Min number of particles
    img_params['attraction_radius'] = max(5, img_params['attraction_radius'])
    img_params['attraction_strength'] = max(0, img_params['attraction_strength'])
    img_params['cell_size'] = max(20, min(200, img_params['cell_size']))

    # --- Randomize Tomography Effects ---
    effects_params = config['effects']
    blur_delta = randint(-4, 4) // 2 * 2 # Ensure change is even to keep kernel odd/even consistent (if needed)
    effects_params['blur_kernel'] += blur_delta
    effects_params['noise_scale'] += uniform(-1000, 1000) # Use uniform for float
    effects_params['noise_correlation_length'] += randint(-1, 1) # Small changes
    effects_params['poisson_noise_factor'] += uniform(-0.1, 0.1)
    effects_params['edge_brightness_factor'] += uniform(-0.1, 0.1)
    effects_params['edge_width'] += randint(-20, 20)
    effects_params['sharpen_amount'] += uniform(-0.2, 0.2)
    # Randomize fractal params slightly
    effects_params['fractal_octaves'] += randint(-1, 1)
    effects_params['fractal_persistence'] += uniform(-0.1, 0.1)

    # --- Randomize NEW Tomography Effects ---
    # Only randomize if simulation is enabled in base config
    if effects_params.get('simulate_projection', False): # Use .get for safety
        effects_params['num_angles'] += randint(-300, 300)
        # Could add randomization for filter choice if desired (e.g., list of filters)
        # ['shepp-logan', 'ram-lak', 'cosine', 'hamming', 'hann']
        # effects_params['reconstruction_filter'] = random.choice(['shepp-logan', 'ram-lak', 'cosine'])
        effects_params['sinogram_blur_sigma'] += uniform(-0.5, 0.5)

    # --- Ensure effect values stay within reasonable bounds ---
    effects_params['blur_kernel'] = max(3, min(15, effects_params['blur_kernel'])) # Post-recon blur likely smaller
    if effects_params['blur_kernel'] % 2 == 0: effects_params['blur_kernel'] +=1 # Ensure blur kernel is odd

    effects_params['noise_scale'] = max(100.0, effects_params['noise_scale'])
    effects_params['noise_correlation_length'] = max(1, effects_params['noise_correlation_length'])
    effects_params['poisson_noise_factor'] = max(0.01, min(5.0, effects_params['poisson_noise_factor'])) # Wider range maybe?
    effects_params['edge_brightness_factor'] = max(1.0, min(1.5, effects_params['edge_brightness_factor']))
    effects_params['edge_width'] = max(10, min(150, effects_params['edge_width']))
    effects_params['sharpen_amount'] = max(0.0, min(1.0, effects_params['sharpen_amount'])) # Sharpen between 0 and 1
    effects_params['fractal_octaves'] = max(1, min(8, effects_params['fractal_octaves']))
    effects_params['fractal_persistence'] = max(0.1, min(0.9, effects_params['fractal_persistence']))

    # Bounds for NEW parameters
    if effects_params.get('simulate_projection', False):
        effects_params['num_angles'] = max(180, min(3600, effects_params['num_angles']))
        effects_params['sinogram_blur_sigma'] = max(0.1, min(5.0, effects_params['sinogram_blur_sigma']))
        # Ensure filter is valid if randomized
        valid_filters = ['shepp-logan', 'ram-lak', 'cosine', 'hamming', 'hann']
        if effects_params['reconstruction_filter'] not in valid_filters:
            effects_params['reconstruction_filter'] = 'shepp-logan' # Default back


    return config


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Generating synthetic XCT slice with multi-label mask...")

    # --- Central Configuration Dictionary ---
    # All parameters are defined here for easy access and modification.
    config = {
        "generation": {
            "output_dir": Path("./synthetic_data_simulated_recon"), # New default dir
            "num_slices": 100,
            "crop_size": 2000, # Final output size after cropping center
            "randomize": True, # Set to False to use exact base parameters
            "plot_single_slice": True # Plot if only one slice is generated
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
            "attraction_strength": 1.0e5, # Adjusted from 0.1e5 for maybe more clustering
            "cell_size": 100 # For spatial grid efficiency
        },
        "effects": {
            # Existing (roles potentially adjusted)
            "blur_kernel": 1, # Post-reconstruction blur kernel size
            "noise_scale": 20000.0, # Correlated noise amplitude
            "noise_correlation_length": 2, # Correlated noise feature size factor
            "poisson_noise_factor": 20, # Factor for sinogram Poisson noise strength
            "edge_brightness_factor": 1.2, # Pre-projection edge effect strength
            "edge_width": 100, # Pre-projection edge effect width
            "sharpen_amount": 0.3, # Post-reconstruction sharpening (0=off, 1=strong basic)
            "fractal_octaves": 6, # Correlated noise complexity
            "fractal_persistence": 0.8, # Correlated noise detail decay

            # New Simulation Parameters
            "simulate_projection": True,       # <<< SET TO True TO ENABLE SIMULATION >>>
            "num_angles": 800,                # Number of projection angles
            "reconstruction_filter": 'shepp-logan', # Recon filter ('ram-lak', 'cosine', etc.)
            "sinogram_blur_sigma": 5,        # Sigma for 1D Gaussian detector blur (pixels)
        }
    }

    # --- Setup Output Directory ---
    output_dir = Path(config['generation']['output_dir']) # Ensure it's Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    num_slices_to_generate = config['generation']['num_slices']

    # --- Generation Loop ---
    for i in range(num_slices_to_generate):
        print(f"\n--- Generating Slice {i+1}/{num_slices_to_generate} ---")

        # Get parameters for this slice (randomized or base)
        if config['generation']['randomize']:
            slice_config = randomize_config(config)
            print("\nUsing Randomized Parameters:")
            # pprint.pprint(slice_config, indent=2, width=100) # Pretty print the config
        else:
            slice_config = deepcopy(config) # Use a copy of the base config
            print("\nUsing Base Parameters:")
            # pprint.pprint(slice_config, indent=2, width=100)

        # --- Create Dataclass Instances from Config ---
        # This provides structure and type hints within the generator classes
        image_params = dataclass_from_dict(ImageParameters, slice_config['image'])
        tomography_effects = dataclass_from_dict(TomographyEffects, slice_config['effects'])

        # --- Generate Slice and Label Image ---
        generator = XCTSliceGenerator(image_params, tomography_effects)
        # generate() now returns the final (potentially reconstructed) image and original label image
        synthetic_slice, label_image = generator.generate()

        # --- Crop Slice and Label Image ---
        crop_size = slice_config['generation']['crop_size']
        image_size = slice_config['image']['size'] # Get size from current config

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


        # --- Plot Results (Optional) ---
        if num_slices_to_generate == 1 and config['generation']['plot_single_slice']:
             print("\nPlotting results...")
             # Plot the final processed slice and the original label map
             plot_results(synthetic_slice_cropped, label_image_cropped)

        # --- Save Results ---
        slice_filename = output_dir / f"synthetic_slice_{i:03d}.tif"
        label_filename = output_dir / f"label_image_{i:03d}.tif" # Use TIF for labels too

        print(f"\nSaving greyscale slice to: {slice_filename}")
        imwrite(slice_filename, synthetic_slice_cropped, imagej=True, metadata={'axes': 'YX'})

        print(f"Saving label image to: {label_filename}")
        # Ensure label image is saved as uint8
        imwrite(label_filename, label_image_cropped.astype(np.uint8), imagej=True, metadata={'axes': 'YX'})

    print(f"\nFinished generating {num_slices_to_generate} slice(s) in '{output_dir}'.")