import os

from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='weather_routing_tool',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='',
    author='52N Developers',
    author_email='info@52north.org',
    description='Python package for weather routing',
    install_requires=[
      'bbox',
      'numpy > 1.23.4',
      'cartopy',
      'geovectorslib',
      'global_land_mask',
      'matplotlib',
      'pandas',
      'pytest',
      'python-dotenv',
      'Pillow',
      'scipy == 1.9.2',
      'setuptools',
      'xarray',
      'netcdf4'
    ]
)
