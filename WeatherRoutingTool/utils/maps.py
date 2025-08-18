class Map:
    lat1: float
    lat2: float
    lon1: float
    lon2: float

    def __init__(self, lat1, lon1, lat2, lon2):
        self.lat1 = float(lat1)
        self.lon1 = float(lon1)
        self.lat2 = float(lat2)
        self.lon2 = float(lon2)
