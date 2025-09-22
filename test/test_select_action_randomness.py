import unittest
import numpy as np
import torch
from collections import Counter

from api import DQNAgent  

STATE_DIM = 8      
ACTION_DIM = 4

class TestSelectActionRandomness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.agent = DQNAgent(STATE_DIM, ACTION_DIM)
        self.zero_state = np.zeros(STATE_DIM, dtype=np.float32)

    def _greedy_index(self):
        # compute the greedy action once for a zero state
        with torch.no_grad():
            s = torch.tensor(self.zero_state, dtype=torch.float32).unsqueeze(0).to(self.agent.device)
            return int(self.agent.policy(s).max(1)[1])

    def test_random_action_distribution_when_epsilon_high(self):
        # Force near-pure random behavior:
        # eps_thresh ≈ 1.0 → take random branch
        self.agent.epsilon = 1.0
        self.agent.e_end = 0.0
        self.agent.e_decay = 10**9

        actions = [self.agent.select_action(self.zero_state) for _ in range(1000)]
        counts = Counter(actions)
        for a in range(ACTION_DIM):
            self.assertTrue(counts[a] > 150, f"Action {a} under-represented: {counts[a]}")

    def test_epsilon_decay_increases_greediness_over_time(self):
        
        greedy = self._greedy_index()

        self.agent.epsilon = 1.0
        self.agent.e_end = 0.01
        self.agent.e_decay = 500

        early = [self.agent.select_action(self.zero_state) for _ in range(500)]
        late  = [self.agent.select_action(self.zero_state) for _ in range(5000)]

        early_greedy_frac = sum(a == greedy for a in early) / len(early)
        late_greedy_frac  = sum(a == greedy for a in late[-1000:]) / 1000

        # loose but meaningful bounds
        self.assertLess(early_greedy_frac, 0.5, f"Early greedy fraction too high: {early_greedy_frac:.2f}")
        self.assertGreater(late_greedy_frac, 0.5, f"Late greedy fraction too low: {late_greedy_frac:.2f}")

if __name__ == "__main__":
    unittest.main()
