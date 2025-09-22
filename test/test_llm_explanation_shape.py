# test_llm_explanation_shape.py
import unittest, numpy as np, pandas as pd, torch, json
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import patch
import api
from api import app, TradingEnv

class DummyLSTM(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.5]], dtype=torch.float32)

class TestLLMExplanationShape(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

        # ---- Minimal OHLCV dataframe (> window_size rows) ----
        n = 30
        close = np.linspace(100, 120, n)
        df = pd.DataFrame({
            "open":   close - 1.0,
            "high":   close + 1.0,
            "low":    close - 2.0,
            "close":  close,
            "volume": np.full(n, 1000, dtype=int),
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        scaler = MinMaxScaler().fit(close.reshape(-1, 1))
        lstm = DummyLSTM()
        env = TradingEnv(df, window_size=10, initial_balance=10_000.0)
        env.reset()  

        sid = "test-sess-123"
        api.sessions[sid] = {
            "env": env,
            "df": df,
            "scaler": scaler,
            "lstm": lstm,
            "agent": None,  
        }
        self.sid = sid

    @patch("api.explain_with_llm", autospec=True)
    def test_llm_explanation_shape(self, mock_explain):
        mock_explain.return_value = {
            "verdict": "BUY",
            "confidence": 0.84,
            "steps": ["Price above SMA(20)", "RSI crossing 50", "Volume rising"],
        }

        payload = {
            "session_id": self.sid,
            "action": 1,          # BUY
            "amount": 500.0,
        }
        resp = self.client.post("/take_action", json=payload)
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))

        body = resp.get_json()
        self.assertIn("llm_explanation", body)
        exp = body["llm_explanation"]

        # Validate shape and types
        for k in ("verdict", "confidence", "steps"):
            self.assertIn(k, exp)
        self.assertIsInstance(exp["verdict"], str)
        self.assertIsInstance(exp["confidence"], (int, float))
        self.assertIsInstance(exp["steps"], list)
        self.assertTrue(all(isinstance(s, str) for s in exp["steps"]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
