import unittest

import numpy

from deep_geometry import GeomScaler

# noinspection PyUnresolvedReferences
from deep_geometry.geom_scaler import localized_mean

dummy_geom = numpy.zeros((1, 1, 5))

square = numpy.array([[
    [0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [0., 1., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
]])

square_duplicate_nodes = numpy.array([[
    [0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [0., 1., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
]])

rectangle = numpy.array([[
    [0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [1., 2., 1., 0., 0.],
    [0., 2., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
]])

normalized_square = numpy.array([[
    [-1., -1., 1., 0., 0.],
    [1., -1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [-1., 1., 1., 0., 0.],
    [-1., -1., 0., 0., 1.],
]])


class TestGeomScaler(unittest.TestCase):
    def test_localized_mean(self) -> None:
        with self.subTest('It rejects inputs other than numpy ndarrays'):
            with self.assertRaises(AssertionError):
                localized_mean(numpy.array([1]))

        with self.subTest('It rejects geometry vectors if it has too many dimensions'):
            with self.assertRaises(AssertionError):
                localized_mean(numpy.zeros((1, 2, 3)))

        with self.subTest('It produces a 1d numpy array of shape (2,)'):
            means = localized_mean(square[0])
            self.assertEqual(numpy.ndim(means), 1)

        with self.subTest('It computes the centroid of a sample square'):
            means = localized_mean(square[0])
            numpy.testing.assert_array_equal(means, 0.5)

    def test_localized_mean_rectangle(self) -> None:
        means = localized_mean(rectangle[0])
        self.assertEqual(means[0], 0.5)
        self.assertEqual(means[1], 1)

    def test_localized_mean_dup_nodes(self) -> None:
        means = localized_mean(square_duplicate_nodes[0])
        numpy.testing.assert_array_equal(means, 0.75)

    def test_scaling_square(self) -> None:
        gs = GeomScaler()
        gs.fit(square)
        self.assertEqual(gs.scale_factor, 0.5)

    def test_scaling_square_dup_nodes(self) -> None:
        gs = GeomScaler()
        gs.fit(square_duplicate_nodes)
        self.assertEqual(gs.scale_factor, 0.5)

    def test_transform(self) -> None:
        with self.subTest('It rejects unfitted geometry scalers on transform'):
            with self.assertRaises(AssertionError):
                gs = GeomScaler()
                gs.transform(square)

        with self.subTest('It transforms a sample square'):
            gs = GeomScaler()
            # scaled_square = square[0] * 2
            # scaled_square[4, 12] = 1.
            gs.fit(square)
            n_square = gs.transform(square)
            numpy.testing.assert_array_equal(n_square, normalized_square)

            coords = [geom[:, :2].flatten() for geom in n_square]
            coords = [item for sublist in coords for item in sublist]
            std = numpy.std(coords)
            numpy.testing.assert_array_almost_equal(std, 1., 1)

    def test_upsized_transform(self) -> None:
        gs = GeomScaler()
        square_0 = square[0] * 2
        square_0[:4, 2] = 1.
        square_0[4, 4] = 1.
        gs.fit(numpy.array([square_0]))
        n_square = gs.transform(numpy.array([square_0]))
        numpy.testing.assert_array_equal(n_square, normalized_square)
        coords = [geom[:, :2].flatten() for geom in n_square]
        coords = [item for sublist in coords for item in sublist]
        std = numpy.std(coords)
        numpy.testing.assert_array_almost_equal(std, 1., 1)

    def test_zero_padded_transform(self) -> None:
        gs = GeomScaler()
        gs.fit(square)
        zero_padding = numpy.repeat(square[:1, -1], repeats=4, axis=0)
        padded_square = numpy.concatenate([square[0], zero_padding], axis=0)
        normalized_square = gs.transform(numpy.array([padded_square]), padding_type='zero')
        numpy.testing.assert_array_equal(normalized_square[0, -4:], zero_padding)

    def test_no_centering(self) -> None:
        gs = GeomScaler()
        gs.fit(square)
        test_square = numpy.copy(square)
        test_square[..., :2] += 2
        gs.scale_factor = 1.  # manually override scale for testing
        normalized_without_centering_square = gs.transform(test_square, with_mean=False)
        numpy.testing.assert_array_equal(test_square, normalized_without_centering_square)

    def test_no_scaling(self) -> None:
        gs = GeomScaler()
        gs.fit(square)
        test_square = numpy.copy(square)
        test_square[..., :2] -= 0.5
        normalized_without_scaling_square = gs.transform(test_square, with_std=False)
        numpy.testing.assert_array_equal(test_square, normalized_without_scaling_square)
