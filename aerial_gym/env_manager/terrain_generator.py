"""
Terrain generation module using Simplex noise.

Generates heightmaps for procedural terrain using multi-octave Simplex noise.
Requires the 'noise' library: pip install noise
"""

import noise
import numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("terrain_generator")


class TerrainGenerator:
    """
    Generates terrain heightmaps using Simplex noise.

    Uses multi-octave Simplex noise to create natural-looking terrain heightmaps.
    """

    def __init__(
        self,
        resolution: int = 256,
        scale_x: float = 20.0,
        scale_y: float = 20.0,
        amplitude: float = 2.0,
        octaves: int = 6,
        frequency: float = 0.1,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        seed=None,
    ):
        self.resolution = resolution
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.amplitude = amplitude
        self.octaves = octaves
        self.frequency = frequency
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        self._heightmap_cache = None
        self._original_seed = self.seed

    def clear_cache(self):
        """Clear the cached heightmap to force regeneration."""
        self._heightmap_cache = None

    def regenerate_with_new_seed(self, seed=None):
        """Clear cache and set a new seed for terrain regeneration."""
        self._heightmap_cache = None
        if seed is not None:
            self.seed = seed
        else:
            import random

            self.seed = random.randint(0, 2**31)
        logger.info(f"TerrainGenerator: Regenerated with seed {self.seed}")

    def generate_heightmap(self, use_cache: bool = True) -> np.ndarray:
        """
        Generate a heightmap using Simplex noise.

        Args:
            use_cache: If True and heightmap already generated, return cached version

        Returns:
            2D numpy array of shape (resolution, resolution) with height values

        """
        if use_cache and self._heightmap_cache is not None:
            return self._heightmap_cache

        heightmap = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        x = np.linspace(0, self.scale_x, self.resolution)
        y = np.linspace(0, self.scale_y, self.resolution)
        X, Y = np.meshgrid(x, y)
        X_norm = X / self.scale_x
        Y_norm = Y / self.scale_y

        noise_value = np.zeros_like(X_norm)

        for octave in range(self.octaves):
            octave_frequency = self.frequency * (self.lacunarity**octave)
            octave_amplitude = self.amplitude * (self.persistence**octave)

            # Offset coordinates by seed to create different noise patterns
            seed_offset_x = (self.seed % 10000) / 1000.0
            seed_offset_y = ((self.seed // 10000) % 10000) / 1000.0
            seed_offset_z = ((self.seed // 100000000) % 10000) / 1000.0 + octave

            for i in range(self.resolution):
                for j in range(self.resolution):
                    noise_value[i, j] += (
                        noise.snoise3(
                            X_norm[i, j] * octave_frequency + seed_offset_x,
                            Y_norm[i, j] * octave_frequency + seed_offset_y,
                            seed_offset_z,
                            octaves=1,
                        )
                        * octave_amplitude
                    )

        # Normalize to [-amplitude/2, amplitude/2] range
        heightmap = noise_value - noise_value.min()
        max_val = heightmap.max()
        if max_val > 0:
            heightmap = heightmap / max_val
            heightmap = (heightmap - 0.5) * self.amplitude
        else:
            logger.warning(f"TerrainGenerator: All noise values are the same (seed: {self.seed})")
            heightmap = np.zeros_like(noise_value)

        if np.any(np.isnan(heightmap)) or np.any(np.isinf(heightmap)):
            logger.error(f"TerrainGenerator: Invalid heightmap values (seed: {self.seed})")
            heightmap = np.zeros_like(noise_value)

        # Cache the heightmap
        self._heightmap_cache = heightmap

        return heightmap

    def sample_height(self, x: float, y: float, heightmap: np.ndarray) -> float:
        """Sample terrain height at a specific (x, y) position."""
        local_x = x + self.scale_x / 2.0
        local_y = y + self.scale_y / 2.0

        i = int((local_y / self.scale_y) * (self.resolution - 1))
        j = int((local_x / self.scale_x) * (self.resolution - 1))
        i = max(0, min(self.resolution - 1, i))
        j = max(0, min(self.resolution - 1, j))

        height = float(heightmap[i, j])
        if np.isnan(height) or np.isinf(height):
            logger.warning(f"TerrainGenerator: Invalid height at ({x}, {y})")
            return 0.0
        return height

    def get_heightmap_bounds(self) -> tuple:
        """
        Get the spatial bounds of the heightmap.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y, min_z, max_z)

        """
        return (0.0, self.scale_x, 0.0, self.scale_y, 0.0, self.amplitude)
