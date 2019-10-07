# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2019-10-07
### Changed
- Renamed `max_points` function to `get_max_points` to avoid confusion with a gotten `max_points` which is a good variable candidate name.
- Refactored the geometry scaler to a class that uses .fit() and .transform() methods, analogous to the scikit-learn scaler objects.
- More consistent names on `geom_wkt` parameter names, to signify passing in a well-known text representation of a geometry.
### Added
- Mypy type hints on all functions, classes and tests
- Better documentation on data normalization usage
- Default support for normalization of replication-padded geometries.

## [1.0.0] - 2018-11-07
### Changed
- First backwards-incompatible release. This is necessary to resolve the loss of information from previous versions on polygon and multipolygon geometries with holes. Previous versions just ignored the holes to keep the encoding format identical to [sketch-rnn](https://github.com/tensorflow/magenta-js/tree/master/sketch) but this project uses much simpler geometries. To include polygon holes without information loss, the format has been extended to include a length 2 one-hot vector for point classes of being part of a inner ring (interior ring or hole) or an outer ring (exterior). This format extension stretches the geometry vector format to shape(num_points, 7).
- The padded output of simplified geometries, or geometry vectors extended to a fixed length are now padded with FULL_STOP bits. This is basically keep yelling to any neural net component to stop interpreting padding.
### Added
- Included an optional `is_last` parameter to the `vectorize_polygon` method of the geovectorizer module. This allows you to specify whether a polygon being vectorized is the last in the (multi-part) geometry or not. Default is false.

## [0.2.0] - 2018-11-06
### Changed
- Fixed a bug where not supplying a get_max_points parameter in the vectorize_wkt function would cause an exception. Fixed in https://github.com/SPINLab/deep-geometry/blob/3f32963a364c60b10aa7fccc5f59468bf3de1dde/deep_geometry/vectorizer.py#L107. Added a regression test.

## [0.1.0] - 2018-09-17
## Added
- First released pypi version on https://pypi.org/project/deep-geometry/: so basically everything