import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn

from grokking import gen_data
from train_checkpoint import save_state
from transf import Transformer


class TrainingArtifactTests(unittest.TestCase):
    def test_split_is_deterministic_and_covers_domain(self):
        def split():
            Tensor.manual_seed(0)
            data, _ = gen_data(63)
            order = Tensor.randperm(len(data)).numpy().astype(np.int64)
            n = int(len(data) * 0.4)
            return data, order[:n], order[n:]

        data, train, test = split()
        _, train_again, test_again = split()
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        self.assertEqual(len(np.unique(np.concatenate((train, test)))), len(data))
        self.assertEqual((len(train), len(test)), (1537, 2307))

    def test_state_roundtrip_preserves_every_model_parameter(self):
        Tensor.manual_seed(0)
        model = Transformer(63, 3, 2, 128, 4, 512)
        state = nn.state.get_state_dict(model)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.npz"
            save_state(state, path)
            restored = np.load(path)
            self.assertEqual(set(restored.files), set(state))
            for key, value in state.items():
                np.testing.assert_array_equal(restored[key], value.numpy())

    def test_source_files_remain_at_recorded_original_hashes(self):
        expected = {
            "grokking.py": "32a946ad312370621e974e0a90c92f5248dfe298d23910dd138f66b5129495d9",
            "transf.py": "97a7e02934f0709d0ce389a44dac3654a5ef9251ccd6035f00d42a4aced83bca",
        }
        for name, digest in expected.items():
            self.assertEqual(
                hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest
            )


if __name__ == "__main__":
    unittest.main()
