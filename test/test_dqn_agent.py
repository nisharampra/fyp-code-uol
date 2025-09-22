import unittest
import numpy as np
import torch
from api import QNetwork, DQNAgent, ReplayBuffer

class TestDQNComponents(unittest.TestCase):

    def setUp(self):
        self.state_dim = 10
        self.action_dim = 3
        self.agent = DQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_qnetwork_forward_shape(self):
        qnet = QNetwork(input_dim=self.state_dim, output_dim=self.action_dim).to(self.device)
        dummy_input = torch.randn(5, self.state_dim).to(self.device)  # batch of 5
        output = qnet(dummy_input)
        self.assertEqual(output.shape, (5, self.action_dim))  # (batch, action_dim)

    def test_select_action(self):
        state = np.random.rand(self.state_dim).astype(np.float32)
        action = self.agent.select_action(state)
        self.assertIn(action, range(self.action_dim))  # Must be one of 0, 1, 2

    def test_learn_updates_q_values(self):
        # Fill replay buffer
        for _ in range(100):
            s = np.random.rand(self.state_dim).astype(np.float32)
            a = np.random.randint(0, self.action_dim)
            r = np.random.rand()
            ns = np.random.rand(self.state_dim).astype(np.float32)
            d = np.random.choice([True, False])
            self.agent.mem.push(s, a, r, ns, d)

        # Run one learning step (no crash = pass)
        self.agent.learn(batch_size=32)

        before = [p.clone() for p in self.agent.policy.parameters()]
        self.agent.learn(batch_size=32)
        after = [p for p in self.agent.policy.parameters()]
        self.assertTrue(any((b != a).any() for b, a in zip(before, after)))

if __name__ == "__main__":
    unittest.main()
