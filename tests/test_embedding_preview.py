"""Verify full-dimensional neighbor scores independently of the 2D projection."""
import numpy as np
import pytest

from lab.embedding_preview import geometry


def test_cosine_uses_every_dimension_and_excludes_vector_magnitude():
    vectors = [[1, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 3], [-1, 0, 0, 0]]
    positions, scores, explained = geometry(vectors)
    np.testing.assert_allclose(scores, [
        [1, 1, 0, -1], [1, 1, 0, -1], [0, 0, 1, 0], [-1, -1, 0, 1]])
    assert positions.shape == (4, 2)
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-12)
    assert explained.sum() == pytest.approx(1)
    np.testing.assert_allclose(geometry(vectors)[0], positions)


def test_projection_reports_information_loss_without_changing_similarity():
    vectors = np.eye(5)
    positions, scores, explained = geometry(vectors)
    np.testing.assert_allclose(scores, vectors)
    assert explained.sum() == pytest.approx(.5)
    assert positions.shape == (5, 2)


@pytest.mark.parametrize("vectors", [
    [[0, 0], [1, 0]], [[float("nan"), 0], [1, 0]], [[1, 0]], [[1, 0], [1]],
])
def test_invalid_embedding_geometry_is_rejected(vectors):
    with pytest.raises(ValueError):
        geometry(vectors)
