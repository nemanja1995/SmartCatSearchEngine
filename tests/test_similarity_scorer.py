import unittest

import numpy as np

from src.SimilarityScorer import SimilarityScorer


class SimilarityScorerCase(unittest.TestCase):
    def test_cosine_similarity(self):
        q = np.array([[1, 1, 2], [2, 2, 1]])
        bs = [[1, 1, 2], [2, 2, 1], [1, 1, 2]]
        base = np.array(bs)

        sim = SimilarityScorer()
        similarity = sim.cosine_similarity(query_vectors=q, corpus_vectors=base)
        similarity = np.round(similarity, 2)
        expected_result = np.array([[1., 0.82],
                                    [0.82, 1.],
                                    [1., 0.82]])
        sim_shape = similarity.shape
        expected_shape = (3, 2)
        self.assertEqual(sim_shape, expected_shape)
        equal = np.array_equal(similarity, expected_result)
        self.assertEqual(True, equal)


if __name__ == '__main__':
    unittest.main()
