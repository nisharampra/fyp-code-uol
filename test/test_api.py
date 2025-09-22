import unittest
import json
from api import app, sessions, trained_models

class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        # Create test client
        self.client = app.test_client()
        self.client.testing = True

    def test_get_tickers(self):
        """GET /tickers should return JSON with tickers list"""
        response = self.client.get("/tickers")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("tickers", data)
        self.assertIsInstance(data["tickers"], list)

    def test_start_session_invalid_ticker(self):
        """POST /start_session with invalid ticker should return 400"""
        response = self.client.post(
            "/start_session",
            data=json.dumps({"ticker": "FAKE"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.data))

    def test_start_session_valid(self):
        """POST /start_session with valid ticker should return session_id"""
        if not trained_models:
            self.skipTest("No trained models available for testing")

        valid_ticker = list(trained_models.keys())[0]
        response = self.client.post(
            "/start_session",
            data=json.dumps({"ticker": valid_ticker}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("session_id", data)
        self.assertIn("initial_state", data)

    def test_state_and_hint_invalid_session(self):
        """GET /state_and_hint with invalid session_id should return 400"""
        response = self.client.get("/state_and_hint?session_id=invalid")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.data))

    def test_take_action_invalid_action(self):
        """POST /take_action with invalid action should return 400"""
        response = self.client.post(
            "/take_action",
            data=json.dumps({"session_id": "invalid", "action": 999}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.data))


if __name__ == "__main__":
    unittest.main()
