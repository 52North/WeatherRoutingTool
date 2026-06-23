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

    def extend_variable(self, var, mode, width):
        if mode == 'min':
            return var - width
        elif mode == 'max':
            return var + width
        else:
            raise ValueError('Only min and max are accepted!')

    def get_widened_map(self, width):
        lat1 = self.extend_variable(self.lat1, 'min', width)
        lat2 = self.extend_variable(self.lat2, 'max', width)
        lon1 = self.extend_variable(self.lon1, 'min', width)
        lon2 = self.extend_variable(self.lon2, 'max', width)
        return Map(lat1, lon1, lat2, lon2)

    def get_var_tuple(self):
        return (self.lat1, self.lon1, self.lat2, self.lon2)
