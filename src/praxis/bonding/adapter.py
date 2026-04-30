"""Adapter: translate ValidatorReport into bond ledger calls."""

from __future__ import annotations

from praxis.bonding._protocol import BondLedger, SlashEvent
from praxis.orchestrator import ValidatorReport
from praxis.protocol.hashing import hash_payload


def justification_hash_for_report(report: ValidatorReport) -> str:
    """Compute a deterministic 64-hex-char content hash for a ValidatorReport.

    The hash covers every field in the canonical signing surface EXCEPT
    generated_at_utc, which is the only non-deterministic field in the schema
    (wall-clock UTC from the validator node; see _models.py for Phase 2 plans).

    Note on validator_version: this field IS included in the canonical surface.
    Two validators running different code versions will compute different
    justification_hash values for the same manifest. Phase 1 accepts this
    divergence; Phase 2 may need to revisit when heterogeneous-version
    validators co-sign slashes via a multi-sig scheme.

    Returns
    -------
    str
        A 64-character lowercase hexadecimal string (blake2b-256 via
        praxis.protocol.hashing.hash_payload).
    """
    payload = report.model_dump(mode="json")
    payload.pop("generated_at_utc", None)
    return hash_payload(payload)


def slash_for_report(
    ledger: BondLedger,
    miner_id: str,
    amount: int,
    report: ValidatorReport,
    justification_url: str,
) -> SlashEvent:
    """Execute a slash on the ledger referencing a ValidatorReport.

    The caller decides when to invoke this function; typically only when
    report.passed is False. The adapter computes the justification hash from
    the report and forwards the slash call to the ledger.

    Belt-and-braces guard: raises ValueError if report.passed is True.
    Slashing on a passed report is almost certainly a bug; the guard surfaces
    it loudly rather than silently minting an invalid slash record.

    Parameters
    ----------
    ledger:
        Any BondLedger implementation.
    miner_id:
        Opaque non-empty identifier for the miner being slashed.
    amount:
        Positive integer amount to slash from miner_id's balance.
    report:
        The ValidatorReport driving this slash. Must have passed=False.
    justification_url:
        URL pointing to the hosted report or evidence. Non-empty string.

    Returns
    -------
    SlashEvent
        The event emitted by the ledger on success.

    Raises
    ------
    ValueError
        If report.passed is True.
    InvalidJustificationError
        If justification_url is empty (forwarded from ledger.slash).
    UnknownMinerError
        If miner_id has no record in the ledger.
    InsufficientBalanceError
        If amount exceeds miner_id's current balance.
    """
    if report.passed:
        raise ValueError(
            f"refusing to slash on a passed report "
            f"(manifest_hash={report.manifest_hash})"
        )
    return ledger.slash(
        miner_id=miner_id,
        amount=amount,
        justification_url=justification_url,
        justification_hash=justification_hash_for_report(report),
    )
