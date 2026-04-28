import base64
import json
from collections.abc import Mapping, Sequence
from hashlib import blake2b
from typing import Any

import numpy as np

_HASH_DIGEST_SIZE = 32


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding suitable for hashing.

    Dicts are emitted with sorted keys, lists keep order, numpy arrays are
    encoded as a tagged dict, and floats are stringified via repr to preserve
    full precision across platforms.
    """
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_payload(payload: Mapping[str, Any]) -> str:
    """blake2b-256 hex digest of canonical_bytes(payload)."""
    return blake2b(canonical_bytes(dict(payload)), digest_size=_HASH_DIGEST_SIZE).hexdigest()


def trajectory_hash(
    observations: Sequence[Any],
    actions: Sequence[Any],
    rewards: Sequence[float],
    terminations: Sequence[bool],
    truncations: Sequence[bool],
    infos: Sequence[Mapping[str, Any]],
    *,
    include_infos: bool = False,
) -> str:
    """blake2b-256 hex digest over a trajectory.

    Infos are dropped from the hash by default since many envs emit
    non-deterministic auxiliary data. Even when include_infos=True, infos are
    only folded into the payload if at least one info dict has content; this
    keeps the hash invariant when every info is empty.
    """
    payload: dict[str, Any] = {
        "observations": list(observations),
        "actions": list(actions),
        "rewards": list(rewards),
        "terminations": list(terminations),
        "truncations": list(truncations),
    }
    if include_infos and any(len(info) > 0 for info in infos):
        payload["infos"] = [dict(info) for info in infos]
    return hash_payload(payload)


def _normalize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__np__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data_b64": base64.b64encode(obj.tobytes()).decode("ascii"),
        }
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": True, "data_b64": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, np.generic):
        return _normalize(obj.item())
    return repr(obj)
