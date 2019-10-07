import re
from typing import List, Optional

import shapely
from shapely import wkt, geometry
import numpy as np
import math

# TODO: refactor GEOMETRY_TYPES to use shapely.geometry.base.GEOMETRY_TYPE
GEOMETRY_TYPES = ["GeometryCollection", "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString",
                  "MultiPolygon", "Geometry"]
X_INDEX = 0  # the X coordinate position
Y_INDEX = 1  # the Y coordinate position
IS_INNER_INDEX = Y_INDEX + 1  # Render index start
IS_OUTER_INDEX = IS_INNER_INDEX + 1
IS_INNER_LEN = 2  # One-hot vector indicating a hole (inner ring) or boundary (outer) ring in a geometry
RENDER_LEN = 3  # Render one-hot vector length
RENDER_INDEX = IS_OUTER_INDEX + 1
ONE_HOT_LEN = 2 + RENDER_LEN  # Length of the one-hot encoded part
STOP_INDEX = RENDER_INDEX + 1  # Stop index for the first geometry. A second one follows
GEO_VECTOR_LEN = STOP_INDEX + 2  # The length needed to describe the features of a geometry point
FULL_STOP_INDEX = -1  # Full stop index. No more points to follow

action_types = ["render", "stop", "full stop"]
wkt_start = {
    "GeometryCollection": " EMPTY",
    "Polygon": "((",
    "MultiPolygon": "(((",
    "Point": "(",
    "LineString": "(",
    "MultiPoint": "((",
    "MultiLineString": "((",
}
wkt_end = {
    "GeometryCollection": "",
    "Polygon": "))",
    "MultiPolygon": ")))",
    "Point": ")",
    "LineString": ")",
    "MultiPoint": "))",
    "MultiLineString": "))",
}


def get_max_points(*wkt_sets: List[str]) -> int:
    """
    Determines the maximum summed size (length) of elements in an arbitrary length 1d array of well-known-text
    geometries
    :param wkt_sets: arbitrary length array of 1d arrays containing well-known-text geometry entries
    :return: scalar integer representing the longest set of points length
    """
    max_points = 0

    for wkts in zip(*wkt_sets):
        number_of_points = sum([num_points_from_wkt(wkt) for wkt in wkts])
        if number_of_points > max_points:
            max_points = number_of_points

    return max_points


def num_points_from_wkt(geom_wkt: str) -> int:
    """
    Calculates the number of points in a well-known text geometry, from a canonical shapely wkt representation
    A 2D point in WKT is a set of two numerical values, separated by a space marked by two decimal values on either side
    :param geom_wkt: a well-known text representation of a geometry
    :return: the number of nodes or points in the geometry
    """
    shape = wkt.loads(geom_wkt)
    pattern = r'\d \d'

    if shape.has_z:
        pattern += r' \d'

    number_of_points = len(re.findall(pattern, shape.wkt))
    return number_of_points


def vectorize_wkt(
        geom_wkt: str,
        max_points: Optional[int] = None,
        simplify: Optional[bool] = False,
        fixed_size: Optional[bool] = False) -> np.ndarray:
    """
    Converts a wkt geometry to a numerical numpy vector representation. The size of the vector is equal to:
        if fixed_size=False: p where p is the size of the set of points in the geometry;
        is fixed_size=True: get_max_points, padded with zeros.
    :param geom_wkt: the geometry as wkt string
    :param max_points: the maximum size of the first output dimension: the maximum number of points
    :param simplify: optional, selecting reduction of points if wkt points exceeds get_max_points
    :param fixed_size: If set to True, the function returns a matrix of size get_max_points
    :return vectors: a 2d numpy array as vectorized representation of the input geometry
    """
    shape = wkt.loads(geom_wkt)
    total_points = num_points_from_wkt(shape.wkt)  # use the shapely wkt form for consistency

    if simplify:
        assert max_points, 'If you want to reduce the number of points using simplify, ' \
                           'please specify the get_max_points.'

    if fixed_size:
        assert max_points, 'If you want to produce fixed sized geometry_vectors, please specify the get_max_points.'

    if max_points and total_points > max_points:
        assert simplify, 'The number of points in the geometry exceeds the get_max_points but the reduce_points ' \
                         'parameter was set to False. Please set the reduce_points parameter to True to reduce ' \
                         'the number of points, or increase get_max_points parameter.'
        shape = recursive_simplify(max_points, shape)
        total_points = num_points_from_wkt(shape.wkt)

    if not max_points:
        max_points = total_points

    if shape.geom_type == 'Polygon':
        geom_matrix = vectorize_polygon(shape, is_last=True)

    elif shape.geom_type == 'MultiPolygon':
        geom_matrix = np.concatenate(
            [vectorize_polygon(geom) for geom in shape.geoms], axis=0)
        geom_matrix[total_points - 1, STOP_INDEX] = 0
        geom_matrix = np.append(geom_matrix, np.zeros((max_points - total_points, GEO_VECTOR_LEN)), axis=0)
        geom_matrix[:total_points - 1, FULL_STOP_INDEX] = 0  # Manually set full stop bits
        geom_matrix[total_points - 1:, FULL_STOP_INDEX] = 1  # Manually set full stop bits

    elif shape.geom_type == 'GeometryCollection':
        if len(shape.geoms) > 0:  # not GEOMETRYCOLLECTION EMPTY
            raise ValueError("Don't know how to process non-empty GeometryCollection type")
        # noinspection PyUnresolvedReferences
        geom_matrix = np.zeros((1, GEO_VECTOR_LEN))
        geom_matrix[:, FULL_STOP_INDEX] = 1  # Manually set full stop bits

    elif shape.geom_type == 'Point':
        geom_matrix = vectorize_points(shape.coords, is_last=True)
    else:
        raise ValueError("Don't know how to get the number of points from geometry type {}".format(shape.geom_type))

    if fixed_size:
        pad_len = max_points - len(geom_matrix)
        pad_shape = ((0, pad_len), (0, 0))
        geom_matrix = np.pad(geom_matrix, pad_shape, mode='constant')
        geom_matrix[:max_points, FULL_STOP_INDEX] = 1
    return geom_matrix


def vectorize_polygon(shape: shapely.geometry, is_last: bool = False) -> np.ndarray:
    """
    Creates a numerical vector from a shapely geometry
    :param shape: the input shapely geometry
    :param is_last: indicates whether this geometry is the last geometry in a collection. Defaults to false.
    :return: an numpy n-dimensional numerical vector representation of the geometry
    """
    if len(shape.interiors):
        vectorized = [vectorize_points(interior.coords, is_inner=True) for interior in shape.interiors]
        vectorized = np.concatenate(vectorized)
        vectorized = np.concatenate([vectorized, vectorize_points(shape.exterior.coords, is_last=is_last)])
    else:
        vectorized = vectorize_points(shape.exterior.coords, is_last=is_last)
    return vectorized


def vectorize_points(points: np.ndarray, is_last: bool = False, is_inner: bool = False) -> np.ndarray:
    """
    Returns a numerical vector representation out of an array of points from a geometry
    :param points: the array of input points
    :param is_last: for the last point in a geometry, to indicate a full stop (true) or a sub-stop (false).
    :param is_inner: if true: sets the IS_INNER one hot vector to one. Denotes that it represents a hole in a geometry.
    :return matrix: a matrix representation of the points.
    """
    number_of_points = len(points)
    matrix = np.zeros((number_of_points, GEO_VECTOR_LEN))

    for point_index, point in enumerate(points):
        matrix[point_index, X_INDEX] = point[0]
        matrix[point_index, Y_INDEX] = point[1]
        if is_inner:
            matrix[point_index, IS_INNER_INDEX] = 1
        else:
            matrix[point_index, IS_OUTER_INDEX] = 1

        if point_index == number_of_points - 1:
            if is_last:
                matrix[point_index, FULL_STOP_INDEX] = True
            else:
                matrix[point_index, STOP_INDEX] = True
        else:
            matrix[point_index, RENDER_INDEX] = True

    return matrix


def recursive_simplify(max_points: int, shape: shapely.geometry) -> shapely.geometry:
    """
    Search algorithm for reducing the number of points of a geometry
    :param max_points:
    :param shape: A shapely shape
    :return:
    """
    log_tolerance: float = -10  # Log scale
    tolerance = math.pow(10, log_tolerance)
    shape = shape.simplify(tolerance)
    while len(re.findall('\d \d', shape.wkt)) > max_points:
        log_tolerance += 0.5
        tolerance = math.pow(10, log_tolerance)
        shape = shape.simplify(tolerance)
    return shape
