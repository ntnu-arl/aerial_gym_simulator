# Procedural Forest Environment

A configurable forest environment with procedurally generated terrain using Simplex noise.

## Features

- **Procedural Terrain Generation**: Multi-octave Simplex noise creates natural-looking terrain heightmaps
- **Tree Density Configuration**: Specify trees per square meter instead of fixed counts
- **Automatic Scaling**: Terrain size automatically matches environment bounds
- **Terrain-Aligned Assets**: Trees are automatically placed on terrain surface

## Configuration

### Environment Setup

```python
from aerial_gym.sim.sim_builder import SimBuilder

env_manager = SimBuilder().build_env(
    env_name="procedural_forest",
    # ... other parameters
)
```

### Key Configuration Parameters

**Environment Bounds** (`procedural_forest_env.py`):
- `lower_bound_min/max`: Lower bounds `[x, y, z]` in meters
- `upper_bound_min/max`: Upper bounds `[x, y, z]` in meters
- Terrain automatically scales to match these bounds

**Tree Density**:
- `tree_density`: Trees per square meter (default: `0.004`)
- Example: `0.004` = 40 trees per 100m × 100m environment
- Tree count is calculated as: `num_trees = tree_density × env_area`

**Terrain Parameters**:
- `terrain_amplitude`: Height range in meters (default: `24.0`)
- `terrain_frequency`: Base frequency (lower=smooth, higher=rough, default: `0.3`)
- `terrain_octaves`: Number of noise layers (default: `8`)
- `terrain_persistence`: Amplitude decay per octave (default: `0.6`)
- `terrain_resolution`: Heightmap resolution (default: `256`)

## Implementation Details

### Files Added/Modified

- **New Environment Config**: `aerial_gym/config/env_config/procedural_forest_env.py`
- **Terrain Generator**: `aerial_gym/env_manager/terrain_generator.py`
- **Tree Config**: Updated `tree_asset_params` in `env_object_config.py` with improved settings
- **Environment Manager**: Added terrain creation and density calculation in `env_manager.py`
- **Asset Manager**: Integrated terrain height sampling for tree placement
- **Isaac Gym Integration**: Heightfield creation in `IGE_env_manager.py`

### Key Design Decisions

1. **Single Tree Config**: Uses existing `tree_asset_params` instead of duplicate config
2. **Automatic Terrain Scaling**: Terrain size derived from environment bounds (no manual `terrain_scale_x/y`)
3. **Density-Based Tree Count**: Trees scale with environment size automatically
4. **Terrain Cache**: Heightmaps cached in `TerrainGenerator` (Isaac Gym heightfields are static)

## Testing

Run the test script to verify the environment:

```bash
python aerial_gym/examples/test_procedural_forest.py
```

This will:
- Create the environment
- Generate segmentation and depth images
- Verify tree placement and terrain generation

## Notes

- Isaac Gym heightfields are static once created (cannot be updated on reset)
- Terrain seed remains constant across resets to keep visual/functional terrain in sync
- Trees are automatically positioned on terrain surface with proper Z offset
- Robot spawns automatically above terrain (minimum 1m clearance)

