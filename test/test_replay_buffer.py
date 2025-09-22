import unittest
import numpy as np
from api import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100)
        self.sample_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def test_push_adds_transition(self):
        self.buffer.push(self.sample_state, 0, 1.0, self.sample_state, False)
        self.assertEqual(len(self.buffer), 1)

    def test_sample_returns_correct_batch_size(self):
        # Add multiple transitions
        for _ in range(30):
            self.buffer.push(self.sample_state, 1, 1.0, self.sample_state, False)
        
        batch_size = 10
        S, A, R, NS, D = self.buffer.sample(batch_size)

        self.assertEqual(S.shape[0], batch_size)
        self.assertEqual(len(A), batch_size)
        self.assertEqual(len(R), batch_size)
        self.assertEqual(NS.shape[0], batch_size)
        self.assertEqual(len(D), batch_size)

if __name__ == "__main__":
    unittest.main()
