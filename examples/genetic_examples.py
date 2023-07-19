import sys
import os
current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(os.path.join(current_path, '..', ''))

from algorithms.routingalg_factory import *

alg = RoutingAlgFactory()
ga = alg.get_routing_alg('GENETIC')
ga.execute_routing()