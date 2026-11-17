"""CPU checks: published weights, affine evidence, plateaus, and box metrics."""
import unittest
import numpy as np
import torch
from model import load_model
from decode import decode_calibrated
from plateaus import representatives
from evaluation import match_frame, metrics


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = load_model()

    def test_checkpoint_and_forward(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 808952)
        with torch.inference_mode():
            out = self.model(torch.full((1, 1, 32, 40), .5))
        self.assertEqual(out['log_size'].shape, (1, 2, 32, 40))
        for value in out.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(self.model.backbone.target_stem_norm.use_population)
        self.assertFalse(self.model.backbone.final[1].use_population)

    def test_constant_evidence(self):
        with torch.inference_mode():
            z, c = self.model.evidence(torch.full((1, 1, 32, 32), .5))
        self.assertEqual(float(z.abs().max()), 0.)
        self.assertEqual(float(c.abs().max()), 0.)

    def test_plateau(self):
        s = np.zeros((5, 5), dtype=np.float32)
        s[1:3, 1:3] = .9
        self.assertEqual(representatives(s, s > .5), [(1, 1)])

    def test_shared_anchor(self):
        main = torch.full((1, 1, 4, 4), -20.)
        main[0, 0, 1, 1] = 5.
        aux = main.clone()
        aux[0, 0, 3, 3] = 10.
        out = dict(main_peak_logits=main, center_logits=.75*main+.25*aux,
                   offsets=torch.zeros(1, 2, 4, 4), log_size=torch.zeros(1, 2, 4, 4))
        anchors = decode_calibrated(out, (1., 1.))[0]['anchors'].tolist()
        self.assertEqual(anchors, [[1, 1]])

    def test_perfect_detection(self):
        r = {'boxes': [[10, 20, 4, 6]], 'group': 'synthetic'}
        f = match_frame(np.array([[10, 20, 14, 26, .9]], dtype=np.float32), r)
        m = metrics([f], .5)
        self.assertEqual(m['AP50'], 1.)
        self.assertEqual(m['F1'], 1.)


if __name__ == '__main__':
    unittest.main()
