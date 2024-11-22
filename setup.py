#this is a setup.py file
from setuptools import setup, find_packages


VERSION = '1.0.8'
DESCRIPTION = 'Python package for hexagonal binning of high-resolution spatial transcriptomic data'
LONG_DESCRIPTION = 'Python package for hexagonal binning of high-resolution spatial transcriptomic data, for more information see https://github.com/BKover99/pseudovisium'


setup(
    name="Pseudovisium",
    version=VERSION,
    author="Bence Kover",
    author_email="<kover.bence@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[
    'numpy',
    'scipy',
    'scanpy',
    'squidpy',
    'matplotlib',
    'opencv-python',
    'h5py',
    'anndata',
    'tifffile',
    'pyarrow',
    'numba',
    'tqdm',
    'pysal',
    'geopandas',
    'seaborn',
    'adjustText',
    'shapely',
    'spatialdata'
        ],

    keywords=['spatial', 'transcriptomics', 'visium', 'xenium', 'cosmx'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X"
    ]
)

