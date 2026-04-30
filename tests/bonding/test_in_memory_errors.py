"""Error-path tests for InMemoryBondLedger."""

from __future__ import annotations

import pytest

from praxis.bonding import (
    InMemoryBondLedger,
    InsufficientBalanceError,
    InvalidJustificationError,
    ReclaimNotSupported,
    UnknownMinerError,
)

_VALID_HASH = "a" * 64
_VALID_URL = "https://example.com/report"


class TestDepositErrors:
    def test_empty_miner_id_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ValueError, match="miner_id must be non-empty"):
            ledger.deposit("", 100)

    def test_zero_amount_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ValueError, match="amount must be positive"):
            ledger.deposit("miner_a", 0)

    def test_negative_amount_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ValueError, match="amount must be positive"):
            ledger.deposit("miner_a", -10)

    def test_failed_deposit_does_not_advance_block(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ValueError):
            ledger.deposit("", 100)
        # No successful op yet; a successful deposit should get block_height=1
        event = ledger.deposit("miner_a", 10)
        assert event.block_height == 1


class TestSlashErrors:
    def test_empty_url_raises_invalid_justification(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        with pytest.raises(InvalidJustificationError, match="justification_url"):
            ledger.slash("miner_a", 10, "", _VALID_HASH)

    def test_bad_hash_short_raises(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        with pytest.raises(InvalidJustificationError, match="64 lowercase hex"):
            ledger.slash("miner_a", 10, _VALID_URL, "abc")

    def test_bad_hash_uppercase_raises(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        with pytest.raises(InvalidJustificationError, match="64 lowercase hex"):
            ledger.slash("miner_a", 10, _VALID_URL, "A" * 64)

    def test_bad_hash_non_hex_raises(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        with pytest.raises(InvalidJustificationError, match="64 lowercase hex"):
            ledger.slash("miner_a", 10, _VALID_URL, "g" * 64)

    def test_empty_miner_id_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ValueError, match="miner_id must be non-empty"):
            ledger.slash("", 10, _VALID_URL, _VALID_HASH)

    def test_zero_amount_raises(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        with pytest.raises(ValueError, match="amount must be positive"):
            ledger.slash("miner_a", 0, _VALID_URL, _VALID_HASH)

    def test_negative_amount_raises(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        with pytest.raises(ValueError, match="amount must be positive"):
            ledger.slash("miner_a", -5, _VALID_URL, _VALID_HASH)

    def test_unknown_miner_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(UnknownMinerError, match="unknown miner"):
            ledger.slash("ghost", 10, _VALID_URL, _VALID_HASH)

    def test_insufficient_balance_raises(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 50)
        with pytest.raises(InsufficientBalanceError, match="exceeds balance"):
            ledger.slash("miner_a", 100, _VALID_URL, _VALID_HASH)

    def test_failed_slash_does_not_advance_block(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 50)
        # Block height is 1 after deposit.
        with pytest.raises(InsufficientBalanceError):
            ledger.slash("miner_a", 100, _VALID_URL, _VALID_HASH)
        # Successful slash should get block_height=2 (not 3).
        event = ledger.slash("miner_a", 10, _VALID_URL, _VALID_HASH)
        assert event.block_height == 2

    def test_failed_slash_does_not_modify_balance(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 50)
        with pytest.raises(InsufficientBalanceError):
            ledger.slash("miner_a", 100, _VALID_URL, _VALID_HASH)
        assert ledger.get_balance("miner_a") == 50


class TestReclaimNotSupported:
    def test_reclaim_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ReclaimNotSupported, match="Phase 2"):
            ledger.reclaim("miner_a", 10, _VALID_URL, _VALID_HASH)

    def test_deny_reclaim_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ReclaimNotSupported, match="Phase 2"):
            ledger.deny_reclaim("miner_a", _VALID_URL)

    def test_can_reclaim_raises(self) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(ReclaimNotSupported, match="Phase 2"):
            ledger.can_reclaim("miner_a", 100)

    def test_reclaim_error_is_bond_ledger_subclass(self) -> None:
        from praxis.bonding import BondLedgerError

        with pytest.raises(BondLedgerError):
            InMemoryBondLedger().reclaim("miner_a", 10, _VALID_URL, _VALID_HASH)
