"""Unit tests for decoding ConditionResolution events from hand-built logs."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from eth_abi import encode
from hexbytes import HexBytes
from web3 import Web3

from src.indexers.polymarket.blockchain import (
    CONDITION_RESOLUTION_TOPIC,
    CONDITIONAL_TOKENS,
    ConditionResolution,
    PolygonClient,
)

CONDITION_ID = "0x" + "11" * 32
QUESTION_ID = "0x" + "22" * 32
ORACLE = Web3.to_checksum_address("0x" + "ab" * 20)
TX_HASH = "0x" + "cd" * 32


def build_log(
    payout_numerators: list[int],
    outcome_slot_count: int | None = None,
    block_number: int = 12345,
    log_index: int = 7,
) -> dict:
    """Build a raw eth_getLogs-style entry for a ConditionResolution event."""
    if outcome_slot_count is None:
        outcome_slot_count = len(payout_numerators)
    data = encode(["uint256", "uint256[]"], [outcome_slot_count, payout_numerators])
    return {
        "address": Web3.to_checksum_address(CONDITIONAL_TOKENS),
        "topics": [
            HexBytes(CONDITION_RESOLUTION_TOPIC),
            HexBytes(CONDITION_ID),
            HexBytes("0x" + "00" * 12 + ORACLE[2:]),
            HexBytes(QUESTION_ID),
        ],
        "data": HexBytes(data),
        "blockNumber": block_number,
        "blockHash": HexBytes("0x" + "ef" * 32),
        "transactionHash": HexBytes(TX_HASH),
        "transactionIndex": 0,
        "logIndex": log_index,
    }


@pytest.fixture()
def client() -> PolygonClient:
    return PolygonClient(rpc_url="http://localhost:8545")


def test_decode_condition_resolution(client: PolygonClient):
    log = build_log([1, 0])

    resolution = client._decode_condition_resolution(log)

    assert resolution.block_number == 12345
    assert resolution.transaction_hash == HexBytes(TX_HASH).hex()
    assert resolution.log_index == 7
    assert resolution.condition_id == CONDITION_ID
    assert resolution.oracle == ORACLE
    assert resolution.question_id == QUESTION_ID
    assert resolution.outcome_slot_count == 2
    assert resolution.payout_numerators == [1, 0]


@pytest.mark.parametrize(
    ("payout_numerators", "expected"),
    [
        ([1, 0], 0),
        ([0, 1], 1),
        ([1, 1], None),  # split/invalid resolution (e.g. 50/50)
        ([0, 0, 1], 2),
        ([2, 3], None),
        ([0, 0], None),
        ([], None),
    ],
)
def test_winning_outcome(payout_numerators: list[int], expected: int | None):
    resolution = ConditionResolution(
        block_number=1,
        transaction_hash="0x00",
        log_index=0,
        condition_id=CONDITION_ID,
        oracle=ORACLE,
        question_id=QUESTION_ID,
        outcome_slot_count=max(len(payout_numerators), 2),
        payout_numerators=payout_numerators,
    )
    assert resolution.winning_outcome == expected


def test_get_condition_resolutions_filters_and_decodes(client: PolygonClient):
    logs = [build_log([1, 0], block_number=100, log_index=1), build_log([0, 1], block_number=200, log_index=2)]
    captured = {}

    def fake_get_logs(params):
        captured.update(params)
        return logs

    with patch.object(client.w3.eth, "get_logs", side_effect=fake_get_logs):
        resolutions = client.get_condition_resolutions(from_block=100, to_block=200)

    assert captured["address"] == Web3.to_checksum_address(CONDITIONAL_TOKENS)
    assert captured["topics"] == [CONDITION_RESOLUTION_TOPIC]
    assert captured["fromBlock"] == 100
    assert captured["toBlock"] == 200
    assert [r.block_number for r in resolutions] == [100, 200]
    assert [r.winning_outcome for r in resolutions] == [0, 1]


def test_get_condition_resolutions_skips_undecodable_logs(client: PolygonClient, capsys: pytest.CaptureFixture):
    bad_log = build_log([1, 0])
    bad_log["data"] = HexBytes("0x")  # truncated payload

    with patch.object(client.w3.eth, "get_logs", return_value=[bad_log, build_log([0, 1])]):
        resolutions = client.get_condition_resolutions(from_block=0, to_block=10)

    assert len(resolutions) == 1
    assert resolutions[0].winning_outcome == 1
    assert "Error decoding log" in capsys.readouterr().out
