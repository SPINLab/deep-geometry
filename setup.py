import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-geometry",
    version="0.2.0",
    author="Rein van 't Veer",
    author_email="rein@geodan.nl",
    description="A python library for preprocessing geospatial vector geometries for use in deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SPINlab/deep-geometry",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'shapely'],
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
