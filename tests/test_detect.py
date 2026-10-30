import unittest

try:
    import numpy as np

    from edgegate import detect
except ImportError:  # numpy and OpenCV: run with .venv-train/bin/python
    detect = None


def squares(xs, size=10.0):
    return np.array([[x, 0.0, x + size, size] for x in xs], np.float32)


@unittest.skipIf(detect is None, "needs OpenCV (.venv-train/bin/python)")
class NmsLimitTest(unittest.TestCase):
    def test_few_boxes_are_plain_per_class_nms(self):
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], np.float32)
        out = detect._select(b, np.array([0.9, 0.8, 0.7], np.float32), np.array([0, 0, 1]), 0.45)
        self.assertEqual([(round(d[4], 3), d[5]) for d in out], [(0.9, 0), (0.7, 1)])

    def test_at_most_max_det_kept_best_first(self):
        n = detect.MAX_DET + 200  # two classes, neither alone over the limit
        scores = np.linspace(0.3, 0.9, n, dtype=np.float32)
        out = detect._select(squares(np.arange(n) * 20.0), scores, np.arange(n) % 2, 0.45)
        self.assertEqual(len(out), detect.MAX_DET)
        self.assertAlmostEqual(min(d[4] for d in out), float(scores[-detect.MAX_DET]), places=6)

    def test_only_the_best_candidates_enter_nms(self):
        # The best MAX_CANDIDATES boxes overlap and one survives; 500 weaker disjoint
        # boxes would all survive if they got in.
        top = detect.MAX_CANDIDATES
        b = np.concatenate([squares(np.zeros(top)), squares(100.0 + np.arange(500) * 20.0)])
        s = np.concatenate([np.linspace(0.9, 0.8, top), np.full(500, 0.5)]).astype(np.float32)
        self.assertEqual(len(detect._select(b, s, np.zeros(len(s), int), 0.45)), 1)


if __name__ == "__main__":
    unittest.main()
