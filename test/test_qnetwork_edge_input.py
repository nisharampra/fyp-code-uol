# test_qnetwork_edge_input.py
import unittest
import torch

# adjust the import to where your QNetwork lives
from api import QNetwork  

STATE_DIM = getattr(__import__("api"), "STATE_DIM", 8)      
ACTION_DIM = getattr(__import__("api"), "ACTION_DIM", 3)    

class TestQNetworkEdgeInput(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = QNetwork(STATE_DIM, ACTION_DIM)
        self.model.eval()  # avoid BatchNorm/Dropout surprises

    def _assert_ok(self, x, batch_size):
        with torch.no_grad():
            y = self.model(x)
        # shape
        self.assertEqual(tuple(y.shape), (batch_size, ACTION_DIM))
        # all finite (no NaN/Inf)
        self.assertTrue(torch.isfinite(y).all().item())

    def test_forward_zero_single(self):
        x = torch.zeros(1, STATE_DIM)
        self._assert_ok(x, 1)

    def test_forward_zero_batch(self):
        x = torch.zeros(16, STATE_DIM)
        self._assert_ok(x, 16)

    def test_forward_large_positive(self):
        x = torch.full((4, STATE_DIM), 1e6)   # very large positive
        self._assert_ok(x, 4)

    def test_forward_large_negative(self):
        x = torch.full((4, STATE_DIM), -1e6)  # very large negative
        self._assert_ok(x, 4)

    def test_forward_mixed_extremes(self):
        # alternating large pos/neg per feature
        v = torch.linspace(-1e6, 1e6, steps=STATE_DIM)
        x = v.repeat(5, 1)                    # batch = 5
        self._assert_ok(x, 5)

if __name__ == "__main__":
    unittest.main()
# ut-28