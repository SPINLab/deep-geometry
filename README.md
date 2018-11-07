# deep-geometry
A python library for preprocessing geospatial vector geometries for use in deep learning

## Rationale
Deep learning can use geospatial vector polygons directly (rather than a feature-extracted pre-processd version), but it requires vectorization and normalisation first, like any data source.

## Installation
`pip install deep-geometry`

## Usage
```python
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

>>> # Just bake me some vectors
... gv.vectorize_wkt(geoms[0])
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  1.]])

>>> gv.vectorize_wkt(geoms[6])
array([[ 0.,  0.,  0., 1., 1.,  0.,  0.],
       [ 1.,  0.,  0., 1., 1.,  0.,  0.],
       [ 1.,  1.,  0., 1., 1.,  0.,  0.],
       [ 0.,  1.,  0., 1., 1.,  0.,  0.],
       [ 0.,  0.,  0., 1., 0.,  0.,  1.]])

>>> # collect the max length from a set of geometries:
... max_len = gv.max_points(geoms)
... print('Maximum geometry node size in set:', max_len)
Maximum geometry node size in set: 7

>>> simple_polygon = gv.vectorize_wkt(geoms[6], max_len)
... print('A polygon:', simple_polygon)
A polygon: [[ 0.  0.  0.  1.  1.  0.  0.]
 [ 1.  0.  0.  1.  1.  0.  0.]
 [ 1.  1.  0.  1.  1.  0.  0.]
 [ 0.  1.  0.  1.  1.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.  1.]]

```