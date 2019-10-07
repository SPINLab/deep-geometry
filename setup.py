import setuptools
import toml

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('Pipfile', 'rt') as p:
    dependencies_toml = p.read()
    dependencies = toml.loads(dependencies_toml)
    dependencies = dependencies['packages']
    dependency_packages = dependencies.keys()


setuptools.setup(
    name="deep-geometry",
    version="2.0.0",
    author="Rein van 't Veer",
    author_email="rein@geodan.nl",
    description="A python library for preprocessing geospatial vector geometries for use in deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SPINlab/deep-geometry",
    packages=setuptools.find_packages(),
    install_requires=dependency_packages,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
