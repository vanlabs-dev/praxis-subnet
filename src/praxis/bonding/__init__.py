"""Bonding shim for Praxis Phase 1.

Public surface mirrors the bactensor/collateral-contracts Solidity interface.
Phase 2 will replace InMemoryBondLedger with a chain-backed adapter.

Exports
-------
BondLedger
    Structural Protocol (runtime_checkable) for any ledger implementation.
InMemoryBondLedger
    Phase 1 reference implementation. deposit, slash, get_balance fully
    implemented. Reclaim path raises ReclaimNotSupported.
DepositEvent, SlashEvent, ReclaimRequest
    Frozen dataclasses returned by the corresponding ledger operations.
BondLedgerError
    Base exception class.
InsufficientBalanceError, UnknownMinerError, InvalidJustificationError
    Specific ledger errors; all inherit from BondLedgerError.
ReclaimNotSupported
    Raised by every reclaim-path call in Phase 1.
justification_hash_for_report
    Compute a deterministic content hash from a ValidatorReport.
slash_for_report
    Execute a slash on a ledger referencing a ValidatorReport.
"""

from praxis.bonding._protocol import (
    BondLedger,
    BondLedgerError,
    DepositEvent,
    InsufficientBalanceError,
    InvalidJustificationError,
    ReclaimNotSupported,
    ReclaimRequest,
    SlashEvent,
    UnknownMinerError,
)
from praxis.bonding.adapter import justification_hash_for_report, slash_for_report
from praxis.bonding.in_memory import InMemoryBondLedger

__all__ = [
    "BondLedger",
    "BondLedgerError",
    "DepositEvent",
    "InMemoryBondLedger",
    "InsufficientBalanceError",
    "InvalidJustificationError",
    "ReclaimNotSupported",
    "ReclaimRequest",
    "SlashEvent",
    "UnknownMinerError",
    "justification_hash_for_report",
    "slash_for_report",
]
