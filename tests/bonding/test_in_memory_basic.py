"""Basic happy-path tests for InMemoryBondLedger."""

from __future__ import annotations

from praxis.bonding import (
    DepositEvent,
    InMemoryBondLedger,
    SlashEvent,
)


class TestDeposit:
    def test_deposit_returns_deposit_event(self) -> None:
        ledger = InMemoryBondLedger()
        event = ledger.deposit("miner_a", 100)
        assert isinstance(event, DepositEvent)

    def test_deposit_event_fields(self) -> None:
        ledger = InMemoryBondLedger()
        event = ledger.deposit("miner_a", 100)
        assert event.miner_id == "miner_a"
        assert event.amount == 100
        assert event.new_balance == 100

    def test_deposit_accumulates(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 50)
        event = ledger.deposit("miner_a", 30)
        assert event.new_balance == 80

    def test_deposit_block_height_increments(self) -> None:
        ledger = InMemoryBondLedger()
        e1 = ledger.deposit("miner_a", 10)
        e2 = ledger.deposit("miner_a", 10)
        assert e1.block_height == 1
        assert e2.block_height == 2


class TestSlash:
    def test_slash_returns_slash_event(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 200)
        event = ledger.slash("miner_a", 50, "https://example.com/report", "a" * 64)
        assert isinstance(event, SlashEvent)

    def test_slash_event_fields(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 200)
        url = "https://example.com/report"
        h = "b" * 64
        event = ledger.slash("miner_a", 50, url, h)
        assert event.miner_id == "miner_a"
        assert event.amount_slashed == 50
        assert event.new_balance == 150
        assert event.justification_url == url
        assert event.justification_hash == h

    def test_slash_block_height_advances(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 200)
        # deposit was block 1
        event = ledger.slash("miner_a", 50, "https://example.com", "c" * 64)
        assert event.block_height == 2

    def test_slash_to_zero_balance(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 100)
        event = ledger.slash("miner_a", 100, "https://example.com", "d" * 64)
        assert event.new_balance == 0
        assert ledger.get_balance("miner_a") == 0


class TestGetBalance:
    def test_unknown_miner_returns_zero(self) -> None:
        ledger = InMemoryBondLedger()
        assert ledger.get_balance("nobody") == 0

    def test_balance_after_deposit(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 300)
        assert ledger.get_balance("miner_a") == 300

    def test_balance_after_slash(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 300)
        ledger.slash("miner_a", 100, "https://example.com", "e" * 64)
        assert ledger.get_balance("miner_a") == 200


class TestMultiMiner:
    def test_balances_are_independent(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("alice", 100)
        ledger.deposit("bob", 200)
        assert ledger.get_balance("alice") == 100
        assert ledger.get_balance("bob") == 200

    def test_slash_on_one_does_not_affect_other(self) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("alice", 100)
        ledger.deposit("bob", 200)
        ledger.slash("alice", 50, "https://example.com", "f" * 64)
        assert ledger.get_balance("alice") == 50
        assert ledger.get_balance("bob") == 200

    def test_block_height_shared_across_miners(self) -> None:
        ledger = InMemoryBondLedger()
        e1 = ledger.deposit("alice", 100)
        e2 = ledger.deposit("bob", 200)
        e3 = ledger.slash("alice", 10, "https://example.com", "0" * 64)
        assert e1.block_height == 1
        assert e2.block_height == 2
        assert e3.block_height == 3
