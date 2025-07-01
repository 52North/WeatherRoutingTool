import warnings

# Filter common warnings in tests
def pytest_configure(config):
    """Configure pytest globally."""
    # Suppress numpy binary incompatibility warnings
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
    
    # Suppress geopandas SQL warnings
    warnings.filterwarnings(
        "ignore", 
        message="pandas only supports SQLAlchemy connectable", 
        module="geopandas"
    )
    
    # Suppress pkg_resources deprecation warnings
    warnings.filterwarnings(
        "ignore", 
        message="pkg_resources is deprecated as an API"
    )
    
    # Suppress cgi deprecation warnings
    warnings.filterwarnings(
        "ignore", 
        message="'cgi' is deprecated", 
        module="webob"
    )
    
    # Suppress CRS warnings from pyogrio
    warnings.filterwarnings(
        "ignore", 
        message="'crs' was not provided", 
        module="pyogrio"
    )
    
    # Suppress geovectorslib warnings
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated as an API.*",
        module="geovectorslib"
    )
