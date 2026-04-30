"""Bond ledger Protocol and event/error types for Praxis Phase 1.

Mirrors the canonical bactensor/collateral-contracts Solidity interface used
by SN12 ComputeHorde and SN51 LIUM in production. Phase 2 will swap the
in-memory implementation for a chain-backed one against the deployed contract.

Error hierarchy
---------------
BondLedgerError (base)
    InsufficientBalanceError   -- slash amount > current balance
    UnknownMinerError          -- miner_id has no record in the ledger
    InvalidJustificationError  -- justification_url or justification_hash fails
                                  format validation
    ReclaimNotSupported        -- reclaim path not implemented in Phase 1

Phase 2 deferrals (explicit)
-----------------------------
- Reclaim path (raises ReclaimNotSupported in Phase 1).
- Justification URL hosting validation (Phase 1 trusts the caller).
- H160 address format for miner_id (Phase 1 uses opaque str).
- Real chain block oracle (Phase 1 uses notional counter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class BondLedgerError(Exception):
    """Base class for all bond ledger errors."""


class InsufficientBalanceError(BondLedgerError):
    """Raised when a slash amount exceeds the miner's current balance."""


class UnknownMinerError(BondLedgerError):
    """Raised when attempting to slash a miner with no ledger record."""


class InvalidJustificationError(BondLedgerError):
    """Raised when the justification URL or hash fails format validation."""


class ReclaimNotSupported(BondLedgerError):
    """Raised by the Phase 1 in-memory ledger for any reclaim-path call.

    Phase 2 chain-backed implementation is required to support reclaim,
    deny_reclaim, and can_reclaim. See bactensor/collateral-contracts.
    """


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DepositEvent:
    """Emitted by BondLedger.deposit() on success."""

    miner_id: str
    amount: int
    new_balance: int
    block_height: int


@dataclass(frozen=True, slots=True)
class SlashEvent:
    """Emitted by BondLedger.slash() on success."""

    miner_id: str
    amount_slashed: int
    new_balance: int
    justification_url: str
    justification_hash: str
    block_height: int


@dataclass(frozen=True, slots=True)
class ReclaimRequest:
    """Returned by BondLedger.reclaim() on success (Phase 2 only).

    Phase 1: this type exists for protocol completeness; the in-memory
    implementation raises ReclaimNotSupported before returning it.
    """

    miner_id: str
    amount: int
    justification_url: str
    justification_hash: str
    block_height: int


# ---------------------------------------------------------------------------
# BondLedger Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BondLedger(Protocol):
    """Structural Protocol for a bond ledger.

    Mirrors the bactensor/collateral-contracts Solidity interface. Implementations
    must provide deposit, slash, reclaim, deny_reclaim, can_reclaim, and get_balance.

    Phase 2 swap: replace InMemoryBondLedger with a chain-backed adapter that
    calls the deployed Solidity contract. The Protocol surface is kept minimal
    so the swap is plug-in.
    """

    def deposit(self, miner_id: str, amount: int) -> DepositEvent:
        """Credit amount to miner_id's balance.

        Raises
        ------
        ValueError
            If miner_id is empty or amount <= 0.
        """
        ...

    def slash(
        self,
        miner_id: str,
        amount: int,
        justification_url: str,
        justification_hash: str,
    ) -> SlashEvent:
        """Debit amount from miner_id's balance with a justification.

        Raises
        ------
        ValueError
            If miner_id is empty or amount <= 0.
        InvalidJustificationError
            If justification_url is empty or justification_hash is not 64
            lowercase hex characters.
        UnknownMinerError
            If miner_id has no record in the ledger.
        InsufficientBalanceError
            If amount exceeds miner_id's current balance.
        """
        ...

    def reclaim(
        self,
        miner_id: str,
        amount: int,
        justification_url: str,
        justification_hash: str,
    ) -> ReclaimRequest:
        """Initiate a reclaim request (Phase 2 only).

        Raises
        ------
        ReclaimNotSupported
            Always in Phase 1.
        """
        ...

    def deny_reclaim(self, miner_id: str, justification_url: str) -> None:
        """Deny a pending reclaim request (Phase 2 only).

        Raises
        ------
        ReclaimNotSupported
            Always in Phase 1.
        """
        ...

    def can_reclaim(self, miner_id: str, current_block: int) -> bool:
        """Check whether a miner is eligible to reclaim (Phase 2 only).

        Raises
        ------
        ReclaimNotSupported
            Always in Phase 1.
        """
        ...

    def get_balance(self, miner_id: str) -> int:
        """Return miner_id's current balance.

        Returns 0 for unknown miners (matches Solidity mapping default).
        Never raises.
        """
        ...


# Suppress unused-import lint for re-export convenience in __init__.py
__all__: list[Any] = [
    "BondLedger",
    "BondLedgerError",
    "DepositEvent",
    "InsufficientBalanceError",
    "InvalidJustificationError",
    "ReclaimNotSupported",
    "ReclaimRequest",
    "SlashEvent",
    "UnknownMinerError",
]
