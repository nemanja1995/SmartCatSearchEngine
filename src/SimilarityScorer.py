"""
Class that can measure similarity between vectors. It uses Cosine similarity.
"""
import numpy as np

# TODO: Fix problem with dividing by zero -> cos_similarity = prod / norms
np.seterr(divide='ignore', invalid='ignore')


class SimilarityScorer:
    def cosine_similarity(self,
                          query_vectors: np.ndarray,
                          corpus_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between question vectors.
        query_vectors is matrix of word vectors dimensionality (e.g. 1000)
        N is the number of questions in the corpus.
        M is number of question for search
        Args:
        query_vector: Vectorized question query of (M, D) shape.
        corpus_vectors: Vectorized question corpus of (N, D) shape.
        Returns:
        The vector of (1, N) shape with values in range [-1, 1] where
        1 is max similarity i.e. two vectors are the same.
        """

        query_vectors_norm = np.linalg.norm(query_vectors, axis=1)
        corpus_vectors_norm = np.linalg.norm(corpus_vectors, axis=1)
        b_corpus_vectors_norm = corpus_vectors_norm[:, np.newaxis]
        b_query_vectors_norm = query_vectors_norm[np.newaxis, :]
        prod = corpus_vectors.dot(query_vectors.T)

        norms = np.multiply(b_corpus_vectors_norm, b_query_vectors_norm)
        # TODO: Fix dividing by zero problem
        cos_similarity = prod / norms
        cos_similarity = np.nan_to_num(cos_similarity)
        return cos_similarity
