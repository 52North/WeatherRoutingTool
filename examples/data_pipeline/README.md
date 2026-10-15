# Data Pipeline Example for Weather Routing Tool

This example demonstrates how to efficiently handle and process weather datasets using scalable techniques.

## Overview

Working with large weather datasets can be memory-intensive and slow when using traditional loading methods. This example shows how to improve performance and scalability by using:

- Chunked data loading with Dask
- Subsetting and interpolation
- Efficient storage formats (NetCDF and Zarr)
- Basic performance comparison

## Features

- Load dataset using chunking (`xarray + Dask`)
- Subset a specific geographic region
- Interpolate data at a given location
- Save processed data in:
  - NetCDF format
  - Zarr format
- Compare execution time for different storage methods

## File Structure
