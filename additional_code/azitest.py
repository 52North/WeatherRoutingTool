from geographiclib.geodesic import Geodesic
from geovectorslib.geod import direct
# The geodesic inverse problem
geod=Geodesic.WGS84
a=geod.Inverse(36.0,10.7,34.5,35.9)
print(a)
b=direct(36,10,34,35)
print(b)
# c=geod.direct(36.0,10.7,34.5,35.9)
# print(c)