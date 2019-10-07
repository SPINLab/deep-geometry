from typing import Optional

import numpy
from deep_geometry.vectorizer import FULL_STOP_INDEX

PADDING_TYPES = ['replication', 'zero']


def localized_mean(geometry_vector: numpy.ndarray) -> numpy.ndarray:
    """
    calculates a centroid for the
    :param geometry_vector:
    :return:
    """
    assert type(geometry_vector) == numpy.ndarray, 'Please provide a numpy ndarray'
    assert numpy.ndim(geometry_vector) == 2, 'Please provide a 2d numpy ndarray'

    full_stop_point_index = get_full_stop_index(geometry_vector)

    # Take the mean of all points before the full stop point for localized origin
    geom_mean = numpy.mean(geometry_vector[0:full_stop_point_index, 0:2], axis=0)
    return numpy.array(geom_mean)


def get_full_stop_index(geometry_vector: numpy.ndarray) -> numpy.ndarray:
    """
    Retrieves the index in the geometry for the first full stop occurrence
    :param geometry_vector: a geometry vector (use vectorize_wkt for this)
    :return: the first full stop occurrence in the geometry
    """
    full_stop_slice = geometry_vector[..., FULL_STOP_INDEX]
    full_stop_point_index = numpy.where(full_stop_slice == 1.)[0][0]

    if full_stop_point_index == 0:
        full_stop_point_index = -1

    return full_stop_point_index


class GeomScaler:
    def __init__(self) -> None:
        self.scale_factor: Optional[float] = None
        self.geom_means = None
        self.min_maxs = None

    def fit(self, geometry_vectors: numpy.ndarray) -> None:
        assert type(geometry_vectors) == numpy.ndarray, 'Please provide a numpy ndarray'
        assert numpy.ndim(geometry_vectors) == 3, 'Please provide a 3d numpy ndarray ' \
                                                  'with axes 0:batch, 1:geometries, 2:points'

        means = [localized_mean(v) for v in geometry_vectors]
        min_maxs = []

        for index, geometry in enumerate(geometry_vectors):
            full_stop_point_index = get_full_stop_index(geometry)

            x_and_y_coords = geometry[:full_stop_point_index, :2]
            min_maxs.append([
                numpy.min(x_and_y_coords - means[index]),
                numpy.max(x_and_y_coords - means[index])
            ])

        self.scale_factor = numpy.std(min_maxs)

    def transform(self,
                  geometry_vectors: numpy.ndarray,
                  padding_type: str = 'replication',
                  with_mean: bool = True,
                  with_std: bool = True
                  ) -> numpy.ndarray:

        assert type(geometry_vectors) == numpy.ndarray, 'Please provide a numpy ndarray'
        assert numpy.ndim(geometry_vectors) == 3, 'Got a vector of rank {}. Please provide a 3d numpy ndarray ' \
                                                  'with axes 0:batch, 1:geometries, 2:points.'\
                                                  .format(numpy.ndim(geometry_vectors))
        assert self.scale_factor, 'Please run the fit() method first before calling this transform method.'
        assert padding_type in PADDING_TYPES, 'Please supply a padding type in {}'.format(PADDING_TYPES)

        localized = numpy.copy(geometry_vectors)
        means = numpy.array([localized_mean(v) for v in geometry_vectors])

        for index, geometry in enumerate(localized):
            if padding_type == 'replication':
                if with_mean:
                    geometry[..., :2] -= means[index]

                if with_std:
                    geometry[..., :2] /= self.scale_factor

            else:  # The only remaining option being zero-padding:
                stop_index = get_full_stop_index(geometry)
                if with_mean:
                    geometry[:stop_index, :2] -= means[index]

                if with_std:
                    geometry[:stop_index, :2] /= self.scale_factor

        return localized
