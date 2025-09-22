# test_replay_buffer_overflow.py
import unittest
import numpy as np
from api import ReplayBuffer

class TestReplayBufferOverflow(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=5)
        # store (state, action, reward, next_state, done)
        for i in range(3):
            s  = np.array([i])
            a  = i
            r  = float(i)
            ns = np.array([i+1])
            d  = False
            self.buffer.push(s, a, r, ns, d)

    def test_sample_more_than_buffer_size(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(10)
        self.assertEqual(len(states), 10)
        self.assertEqual(len(actions), 10)
        self.assertEqual(len(rewards), 10)
        self.assertEqual(len(next_states), 10)
        self.assertEqual(len(dones), 10)

if __name__ == "__main__":
    unittest.main()
