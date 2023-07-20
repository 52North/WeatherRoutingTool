import os

from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='WeatherRoutingTool',
    version='0.1',
    include_package_data=True,
    packages=find_packages(),
    url='',
    license='',
    author='52N Developers',
    author_email='info@52north.org',
    description='Python package for weather routing',
    install_requires=[
      'bbox',
      'dask',
      'geovectorslib',
      'global_land_mask',
      'lxml',
      'matplotlib',
      'numpy == 1.23.4',
      'pandas',
      'pytest',
      'Pillow',
      'scipy == 1.9.2',
      'setuptools',
      'xarray',
      'netcdf4'
    ]
)
os.system('pip install wheel')
os.system('pip install git+https://github.com/52North/MariGeoRoute#subdirectory=data/maridatadownloader')
os.system('pip install cartopy')
