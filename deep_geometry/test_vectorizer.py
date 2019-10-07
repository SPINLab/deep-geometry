import unittest

import numpy as np
from shapely import wkt as wktreader
from csv import DictReader

from deep_geometry.vectorizer \
    import num_points_from_wkt, vectorize_wkt, get_max_points, \
    GEO_VECTOR_LEN, IS_INNER_INDEX, IS_OUTER_INDEX, FULL_STOP_INDEX

TOPOLOGY_CSV = 'test_files/polygon_multipolygon.csv'

SOURCE_DATA = []
with open(TOPOLOGY_CSV, 'r') as csv_file:
    reader = DictReader(csv_file, )
    for row in reader:
        SOURCE_DATA.append(row)

brt_wkt = [record['brt_wkt'] for record in SOURCE_DATA]
osm_wkt = [record['osm_wkt'] for record in SOURCE_DATA]
target_wkt = [record['intersection_wkt'] for record in SOURCE_DATA]

input_geom = np.array([
    [0., 0., 1., 0., 0.],
    [0., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0.],
    [0., -1., 1., 0., 0.],
    [-1., -1., 1., 0., 0.],
    [-1., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0.]
])

output_geom = np.array([
    [0.0, 0.00, 1., 0., 0.],
    [0.0, 0.25, 1., 0., 0.],
    [0.0, 0.50, 1., 0., 0.],
    [0.0, 0.75, 1., 0., 0.],
    [0.0, 1.00, 1., 0., 0.],
    [0.25, 1.0, 1., 0., 0.],
    [0.50, 1.0, 1., 0., 0.],
    [1.0, 1.00, 1., 0., 0.],
    [1.0, 0.50, 1., 0., 0.],
    [1.0, 0.00, 1., 0., 0.],
    [0.5, 0.00, 1., 0., 0.],
    [0.0, 0.00, 0., 1., 0.],
    [0.0, 0.00, 1., 0., 0.],
    [0.0, -0.5, 1., 0., 0.],
    [0.0, -1.0, 1., 0., 0.],
    [-0.5, -1., 1., 0., 0.],
    [-1., -1.0, 1., 0., 0.],
    [-1., -0.5, 1., 0., 0.],
    [-1., 0.00, 1., 0., 0.],
    [-0.5, 0.0, 1., 0., 0.],
    [0.00, 0.0, 0., 0., 1.],
    [0.00, 0.0, 0., 0., 0.]
])

non_empty_geom_collection = 'GEOMETRYCOLLECTION(LINESTRING(1 1, 3 5),POLYGON((-1 -1, -1 -5, -5 -5, -5 -1, -1 -1)))'


class TestVectorizer(unittest.TestCase):
    def test_num_points_from_multipolygon_with_hole(self) -> None:
        with open('test_files/multipolygon_with_hole.txt', 'r') as file:
            wkt = file.read()
            total_points = num_points_from_wkt(wkt)
            self.assertEqual(total_points, 683)

    def test_max_points(self) -> None:
        max_pts = get_max_points(brt_wkt, osm_wkt)
        self.assertEqual(max_pts, 159)

    def test_max_points_3d(self) -> None:
        geom_3d = 'POLYGON((0 0 0, 1 1 1, 2 2 2, 0 0 0))'
        max_pts = get_max_points([geom_3d])
        self.assertEqual(max_pts, 4)

    def test_vectorize_one_wkt(self) -> None:
        max_points = 20
        input_set = target_wkt
        vectorized = []
        for index in range(len(input_set)):
            vectorized.append(vectorize_wkt(input_set[index], max_points, simplify=True))
        self.assertEqual(len(input_set), len(brt_wkt))
        self.assertEqual(vectorized[0].shape, (19, GEO_VECTOR_LEN))
        self.assertEqual(vectorized[1].shape, (1, GEO_VECTOR_LEN))

    def test_no_max_points_fixed_size(self) -> None:
        input_set = np.array(target_wkt)
        with self.assertRaises(AssertionError):
            vectorized = [vectorize_wkt(wkt, fixed_size=True) for wkt in input_set]

    def test_fixed_size(self) -> None:
        max_points = 20
        input_set = np.array(target_wkt)
        vectorized = [vectorize_wkt(wkt, max_points, simplify=True, fixed_size=True) for wkt in input_set]
        self.assertEqual(np.array(vectorized).shape, (input_set.size, 20, GEO_VECTOR_LEN))
        for vector in vectorized:
            self.assertEqual(vector[-1, FULL_STOP_INDEX], 1)

    def test_non_empty_geom_coll(self) -> None:
        with self.assertRaises(ValueError):
            vectorize_wkt(non_empty_geom_collection, 100)

    def test_point_without_max_points(self) -> None:
        vectorized = vectorize_wkt('POINT(12 14)')
        self.assertEqual(vectorized.shape, (1, GEO_VECTOR_LEN))
        self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)  # Test full stop bit

    def test_point_with_max_points(self) -> None:
        vectorized = vectorize_wkt('POINT(12 14)', 5)
        self.assertEqual(vectorized.shape, (1, GEO_VECTOR_LEN))
        self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)  # Test full stop bit

    def test_unsupported_geom(self) -> None:
        with self.assertRaises(Exception):
            vectorize_wkt(
                'THIS_SHOULD_THROW_AN_EXCEPTION ((10 10, 20 20, 10 40),(40 40, 30 30, 40 20, 30 10))', 16)

    def test_multipolygon(self) -> None:
        with open('test_files/multipolygon.txt', 'r') as file:
            wkt = file.read()
            vectorized = vectorize_wkt(wkt)
            self.assertEqual((333, GEO_VECTOR_LEN), vectorized.shape)
            self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)  # Test full stop bit

    def test_polygon_with_hole(self) -> None:
        polygon_with_hole = "POLYGON((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))"
        vectorized = vectorize_wkt(polygon_with_hole)
        for is_inner_bit in vectorized[:5, IS_INNER_INDEX]:
            self.assertEqual(is_inner_bit, 1)
        for is_outer_bit in vectorized[5:, IS_OUTER_INDEX]:
            self.assertEqual(is_outer_bit, 1)
        self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)

    def test_multipolygon_with_hole(self) -> None:
        with open('test_files/multipolygon_with_hole.txt', 'r') as file:
            wkt = file.read()
            vectorized = vectorize_wkt(wkt)
            self.assertEqual((683, GEO_VECTOR_LEN), vectorized.shape)
            self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)  # Test full stop bit

    def test_vectorize_big_multipolygon(self) -> None:
        with open('test_files/big_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_pts = get_max_points([wkt])
            vectorized = vectorize_wkt(wkt, max_pts)
            self.assertEqual((144, GEO_VECTOR_LEN), vectorized.shape)
            self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)  # Test full stop bit

    def test_simplify_multipolygon_gt_max_points(self) -> None:
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 20
            vectorized = vectorize_wkt(wkt, max_points, simplify=True)
            self.assertEqual((20, GEO_VECTOR_LEN), vectorized.shape)
            self.assertEqual(vectorized[-1, FULL_STOP_INDEX], 1)  # Test full stop bit

    def test_simplify_without_max_points(self) -> None:
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            with self.assertRaises(AssertionError):
                vectorize_wkt(wkt, simplify=True)

    def test_multipolygon_exceed_max_points(self) -> None:
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 20
            with self.assertRaises(Exception):
                vectorize_wkt(wkt, max_points)

    def test_polygon_exceed_max_points(self) -> None:
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            shape = wktreader.loads(wkt)
            geom = shape.geoms[0]
            max_points = 20
            with self.assertRaises(Exception):
                vectorize_wkt(geom.wkt, max_points)
