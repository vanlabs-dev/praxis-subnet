"""Hypothesis property tests for InMemoryBondLedger.

Four properties:
1. Conservation law: sum(deposits) - sum(slashes) == sum(balances).
2. Slash never exceeds balance (invariant after every successful slash).
3. Block monotonicity: successful operations yield strictly increasing heights.
4. Miner independence: slashing one miner does not change another's balance.

Design note: tests track totals externally (do not access private attributes)
and rely only on the public API (deposit, slash, get_balance). get_balance is
exercised as a side-effect of the conservation check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from praxis.bonding import (
    InMemoryBondLedger,
    InsufficientBalanceError,
    InvalidJustificationError,
    UnknownMinerError,
)

# ---------------------------------------------------------------------------
# Operation model
# ---------------------------------------------------------------------------

_MINERS = [f"miner_{i}" for i in range(3)]
_VALID_HASH_ALPHABET = "0123456789abcdef"


@dataclass
class _Op:
    kind: Literal["deposit", "slash"]
    miner_id: str
    amount: int
    url: str
    hash_: str


def _op_strategy() -> st.SearchStrategy[_Op]:  # type: ignore[type-arg]
    miner = st.sampled_from(_MINERS)
    amount = st.integers(min_value=1, max_value=500)
    url = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/:.-_"),
        min_size=1,
        max_size=50,
    )
    hash_ = st.text(alphabet=_VALID_HASH_ALPHABET, min_size=64, max_size=64)
    kind = st.sampled_from(["deposit", "slash"])
    return st.builds(_Op, kind=kind, miner_id=miner, amount=amount, url=url, hash_=hash_)


# ---------------------------------------------------------------------------
# Property 1: Conservation law
# ---------------------------------------------------------------------------


@given(operations=st.lists(_op_strategy(), min_size=0, max_size=20))
@settings(max_examples=200, deadline=None)
def test_conservation_law(operations: list[_Op]) -> None:  # type: ignore[misc]
    """sum(deposits) - sum(slashes) == sum(balances) after any op sequence."""
    ledger = InMemoryBondLedger()
    expected_deposited: dict[str, int] = {m: 0 for m in _MINERS}
    expected_slashed: dict[str, int] = {m: 0 for m in _MINERS}

    for op in operations:
        try:
            if op.kind == "deposit":
                ledger.deposit(op.miner_id, op.amount)
                expected_deposited[op.miner_id] += op.amount
            elif op.kind == "slash":
                ledger.slash(op.miner_id, op.amount, op.url, op.hash_)
                expected_slashed[op.miner_id] += op.amount
        except (UnknownMinerError, InsufficientBalanceError, InvalidJustificationError):
            continue  # expected; does not violate conservation

    expected_balance_sum = sum(expected_deposited.values()) - sum(expected_slashed.values())
    actual_balance_sum = sum(ledger.get_balance(m) for m in _MINERS)
    assert actual_balance_sum == expected_balance_sum


# ---------------------------------------------------------------------------
# Property 2: Slash never exceeds balance
# ---------------------------------------------------------------------------


@given(operations=st.lists(_op_strategy(), min_size=1, max_size=20))
@settings(max_examples=200, deadline=None)
def test_slash_never_exceeds_balance(operations: list[_Op]) -> None:  # type: ignore[misc]
    """get_balance is always >= 0 after every successful operation."""
    ledger = InMemoryBondLedger()

    for op in operations:
        try:
            if op.kind == "deposit":
                ledger.deposit(op.miner_id, op.amount)
            elif op.kind == "slash":
                ledger.slash(op.miner_id, op.amount, op.url, op.hash_)
        except (UnknownMinerError, InsufficientBalanceError, InvalidJustificationError):
            pass

        for m in _MINERS:
            assert ledger.get_balance(m) >= 0


# ---------------------------------------------------------------------------
# Property 3: Block monotonicity for successful ops
# ---------------------------------------------------------------------------


@given(operations=st.lists(_op_strategy(), min_size=1, max_size=20))
@settings(max_examples=200, deadline=None)
def test_block_monotonicity(operations: list[_Op]) -> None:  # type: ignore[misc]
    """Block heights returned by successful operations are strictly increasing."""
    ledger = InMemoryBondLedger()
    heights: list[int] = []

    for op in operations:
        try:
            if op.kind == "deposit":
                event = ledger.deposit(op.miner_id, op.amount)
                heights.append(event.block_height)
            elif op.kind == "slash":
                event2 = ledger.slash(op.miner_id, op.amount, op.url, op.hash_)
                heights.append(event2.block_height)
        except (UnknownMinerError, InsufficientBalanceError, InvalidJustificationError):
            pass  # failed ops don't contribute a height

    # Every consecutive pair must be strictly increasing.
    for i in range(len(heights) - 1):
        assert heights[i] < heights[i + 1]


# ---------------------------------------------------------------------------
# Property 4: Miner independence
# ---------------------------------------------------------------------------


@given(
    deposit_amounts=st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=5),
    slash_amount=st.integers(min_value=1, max_value=500),
    hash_=st.text(alphabet=_VALID_HASH_ALPHABET, min_size=64, max_size=64),
)
@settings(max_examples=200, deadline=None)
def test_miner_independence(  # type: ignore[misc]
    deposit_amounts: list[int],
    slash_amount: int,
    hash_: str,
) -> None:
    """Slashing miner_0 does not change miner_1's balance."""
    ledger = InMemoryBondLedger()
    m0, m1 = "miner_0", "miner_1"

    for amt in deposit_amounts:
        ledger.deposit(m0, amt)
        ledger.deposit(m1, amt)

    balance_m1_before = ledger.get_balance(m1)

    try:
        ledger.slash(m0, slash_amount, "https://example.com", hash_)
    except (InsufficientBalanceError, UnknownMinerError, InvalidJustificationError):
        pass  # slash may fail; m1 balance should still be unchanged

    assert ledger.get_balance(m1) == balance_m1_before
