# deep-geometry
A python library for preprocessing geospatial vector geometries for use in deep learning

## Rationale
Deep learning can use geospatial vector polygons directly (rather than a feature-extracted pre-processd version), but it requires vectorization and normalisation first, like any data source.

## Installation
`pip install deep-geometry`

## Usage
### Geometry vectorization
Make a numerical vector from a geometry: 
```
>>> from deep_geometry import vectorizer as gv

>>> geoms = [
...     'POINT(0 0)',
...     'POINT(1 1)',
...     'POINT(2 2)',
...     'POINT(3 3)',
...     'POINT(4 4)',
...     'POINT(5 5)',
...     'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
... ]

>>> gv.vectorize_wkt(geoms[0])
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  1.]])

>>> gv.vectorize_wkt(geoms[6])
array([[ 0.,  0.,  0., 1., 1.,  0.,  0.],
       [ 1.,  0.,  0., 1., 1.,  0.,  0.],
       [ 1.,  1.,  0., 1., 1.,  0.,  0.],
       [ 0.,  1.,  0., 1., 1.,  0.,  0.],
       [ 0.,  0.,  0., 1., 0.,  0.,  1.]])
```

Collect the max length from a set of geometries:
```
>>> max_len = gv.get_max_points(geoms)
>>> print('Maximum geometry node size in set:', max_len)
Maximum geometry node size in set: 7
```

### Numerical data normalization
Geometries regularly are in some kind of earth projection that is far from the origin of the coordinate system. In order for machine learning models to learn, data needs to be normalized. A usual way to go about this is to mean-center the instances and to divide by the dataset standard deviation.

The library provides a convenience class for normalization, modeled after the scalers from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) with a .fit() and a .transform() method:

```
>>> from deep_geometry import GeomScaler
>>> import numpy
>>> gs = GeomScaler()  # simply initialize
>>> geom6 = gv.vectorize_wkt(geoms[6])
>>> dataset = numpy.repeat([geom6], 4, axis=0)
>>> gs.fit(dataset)
>>> gs.scale_factor
0.5
>>> normalized_data = gs.transform(dataset)
>>> normalized_data[0]  # see: zero-mean and scaled to standard deviation
array([[-1., -1.,  0.,  1.,  1.,  0.,  0.],
       [ 1., -1.,  0.,  1.,  1.,  0.,  0.],
       [ 1.,  1.,  0.,  1.,  1.,  0.,  0.],
       [-1.,  1.,  0.,  1.,  1.,  0.,  0.],
       [-1., -1.,  0.,  1.,  0.,  0.,  1.]])

``` 
