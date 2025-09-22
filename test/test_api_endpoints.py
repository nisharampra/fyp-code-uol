import requests
import pytest

BASE_URL = "http://localhost:5001"

# Store session_id across tests
session_id = None


def test_UT10_get_tickers():
    response = requests.get(f"{BASE_URL}/tickers")
    assert response.status_code == 200
    data = response.json()
    assert "tickers" in data
    assert isinstance(data["tickers"], list)
    assert len(data["tickers"]) > 0


def test_UT11_start_session():
    global session_id
    tickers = requests.get(f"{BASE_URL}/tickers").json()["tickers"]
    response = requests.post(f"{BASE_URL}/start_session", json={"ticker": tickers[0]})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "initial_state" in data
    session_id = data["session_id"]


def test_UT12_state_and_hint():
    global session_id
    assert session_id is not None
    response = requests.get(f"{BASE_URL}/state_and_hint", params={"session_id": session_id})
    assert response.status_code == 200
    data = response.json()
    assert "hint" in data
    assert "balance" in data
    assert "shares_held" in data


def test_UT13_take_action():
    global session_id
    assert session_id is not None
    payload = {"session_id": session_id, "action": 0}
    response = requests.post(f"{BASE_URL}/take_action", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "next_state" in data
    assert "reward" in data
    assert "done" in data
    assert "correct" in data


def test_UT14_invalid_inputs():
    # Invalid ticker
    bad_ticker = requests.post(f"{BASE_URL}/start_session", json={"ticker": "FAKE123"})
    assert bad_ticker.status_code == 400

    # Invalid session for /state_and_hint
    bad_state = requests.get(f"{BASE_URL}/state_and_hint", params={"session_id": "bad-id"})
    assert bad_state.status_code == 400

    # Invalid session for /take_action
    bad_action = requests.post(f"{BASE_URL}/take_action", json={"session_id": "bad-id", "action": 1})
    assert bad_action.status_code == 400

    # Valid ticker, invalid action
    tickers = requests.get(f"{BASE_URL}/tickers").json()["tickers"]
    response = requests.post(f"{BASE_URL}/start_session", json={"ticker": tickers[0]})
    sid = response.json()["session_id"]
    invalid_action = requests.post(f"{BASE_URL}/take_action", json={"session_id": sid, "action": 99})
    assert invalid_action.status_code == 400
