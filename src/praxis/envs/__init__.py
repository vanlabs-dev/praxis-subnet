"""Reference gymnasium environments for the Praxis subnet."""

import gymnasium as gym

from praxis.envs.gridworld import PraxisGridworld

__all__ = ["PraxisGridworld"]

# Register three difficulty instances of PraxisGridworld.
# max_episode_steps is NOT passed here -- the env owns truncation via its
# internal counter so the determinism check sees a single source of truth.
# Gymnasium's TimeLimit wrapper would shadow that logic if we passed it here.
_REGISTRATIONS: list[tuple[str, int]] = [
    ("PraxisGridworld-Easy-v0", 5),
    ("PraxisGridworld-Medium-v0", 10),
    ("PraxisGridworld-Hard-v0", 20),
]

for _env_id, _grid_size in _REGISTRATIONS:
    if _env_id not in gym.envs.registry:
        gym.register(
            id=_env_id,
            entry_point="praxis.envs.gridworld:PraxisGridworld",
            kwargs={"grid_size": _grid_size},
        )

# Lowercase aliases satisfy the EnvManifest env_id pattern (^[a-z][a-z0-9_-]{2,63}$)
# while mapping to the same underlying environments. Required for the determinism
# validator which calls gym.make(manifest.env_id).
_LOWERCASE_ALIASES: list[tuple[str, int]] = [
    ("praxisgridworld-easy-v0", 5),
    ("praxisgridworld-medium-v0", 10),
    ("praxisgridworld-hard-v0", 20),
]

for _env_id, _grid_size in _LOWERCASE_ALIASES:
    if _env_id not in gym.envs.registry:
        gym.register(
            id=_env_id,
            entry_point="praxis.envs.gridworld:PraxisGridworld",
            kwargs={"grid_size": _grid_size},
        )
