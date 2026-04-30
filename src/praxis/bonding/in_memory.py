"""Phase 1 in-memory reference implementation of BondLedger."""

from __future__ import annotations

import re

from praxis.bonding._protocol import (
    BondLedger,
    DepositEvent,
    InsufficientBalanceError,
    InvalidJustificationError,
    ReclaimNotSupported,
    ReclaimRequest,
    SlashEvent,
    UnknownMinerError,
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECLAIM_PHASE_2_MESSAGE = (
    "Phase 1 in-memory ledger does not implement reclaim path; "
    "Phase 2 chain-backed implementation required "
    "(see bactensor/collateral-contracts)."
)


class InMemoryBondLedger:
    """Phase 1 reference implementation of BondLedger.

    Mirrors the bactensor/collateral-contracts Solidity interface for deposit,
    slash, and get_balance. Reclaim path raises ReclaimNotSupported with an
    explicit Phase 2 marker message.

    Conservation law: for any sequence of successful operations,
        sum(total_deposited.values()) - sum(total_slashed.values())
            == sum(balances.values())

    Block height: notional integer counter, increments ONLY on SUCCESSFUL
    state-mutating calls. Failed operations (those raising exceptions) leave
    the counter unchanged, matching Solidity revert semantics.
    """

    def __init__(self) -> None:
        self._balances: dict[str, int] = {}
        self._total_deposited: dict[str, int] = {}
        self._total_slashed: dict[str, int] = {}
        self._current_block: int = 0

    def deposit(self, miner_id: str, amount: int) -> DepositEvent:
        """Credit amount to miner_id's balance.

        Raises
        ------
        ValueError
            If miner_id is empty or amount <= 0.
        """
        if not miner_id:
            raise ValueError("miner_id must be non-empty")
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._balances[miner_id] = self._balances.get(miner_id, 0) + amount
        self._total_deposited[miner_id] = self._total_deposited.get(miner_id, 0) + amount
        self._current_block += 1
        return DepositEvent(
            miner_id=miner_id,
            amount=amount,
            new_balance=self._balances[miner_id],
            block_height=self._current_block,
        )

    def slash(
        self,
        miner_id: str,
        amount: int,
        justification_url: str,
        justification_hash: str,
    ) -> SlashEvent:
        """Debit amount from miner_id's balance with a justification.

        Justification validation runs before miner and amount validation so that
        callers receive clear errors about malformed inputs regardless of ledger
        state.

        Raises
        ------
        InvalidJustificationError
            If justification_url is empty or justification_hash is not 64
            lowercase hex characters.
        ValueError
            If miner_id is empty or amount <= 0.
        UnknownMinerError
            If miner_id has no record in the ledger.
        InsufficientBalanceError
            If amount exceeds miner_id's current balance.
        """
        if not justification_url:
            raise InvalidJustificationError("justification_url must be non-empty")
        if not _HASH_PATTERN.match(justification_hash):
            raise InvalidJustificationError(
                "justification_hash must be 64 lowercase hex chars"
            )
        if not miner_id:
            raise ValueError("miner_id must be non-empty")
        if amount <= 0:
            raise ValueError("amount must be positive")
        if miner_id not in self._balances:
            raise UnknownMinerError(f"unknown miner: {miner_id}")
        if amount > self._balances[miner_id]:
            raise InsufficientBalanceError(
                f"slash amount {amount} exceeds balance {self._balances[miner_id]} "
                f"for miner {miner_id}"
            )
        self._balances[miner_id] -= amount
        self._total_slashed[miner_id] = self._total_slashed.get(miner_id, 0) + amount
        self._current_block += 1
        return SlashEvent(
            miner_id=miner_id,
            amount_slashed=amount,
            new_balance=self._balances[miner_id],
            justification_url=justification_url,
            justification_hash=justification_hash,
            block_height=self._current_block,
        )

    def reclaim(
        self,
        miner_id: str,
        amount: int,
        justification_url: str,
        justification_hash: str,
    ) -> ReclaimRequest:
        """Not implemented in Phase 1.

        Raises
        ------
        ReclaimNotSupported
            Always. Phase 2 chain-backed implementation required.
        """
        raise ReclaimNotSupported(_RECLAIM_PHASE_2_MESSAGE)

    def deny_reclaim(self, miner_id: str, justification_url: str) -> None:
        """Not implemented in Phase 1.

        Raises
        ------
        ReclaimNotSupported
            Always. Phase 2 chain-backed implementation required.
        """
        raise ReclaimNotSupported(_RECLAIM_PHASE_2_MESSAGE)

    def can_reclaim(self, miner_id: str, current_block: int) -> bool:
        """Not implemented in Phase 1.

        Raises
        ------
        ReclaimNotSupported
            Always. Phase 2 chain-backed implementation required.
        """
        raise ReclaimNotSupported(_RECLAIM_PHASE_2_MESSAGE)

    def get_balance(self, miner_id: str) -> int:
        """Return miner_id's current balance.

        Returns 0 for unknown miners, matching Solidity mapping default.
        """
        return self._balances.get(miner_id, 0)


# Satisfy BondLedger structural Protocol at import time.
_: BondLedger = InMemoryBondLedger()
del _
