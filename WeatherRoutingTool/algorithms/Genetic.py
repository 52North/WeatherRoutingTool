from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
import numpy as np
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from GeneticUtils import *

class Population(Sampling):
    def __init__(self, src, dest, X,var_type=np.float64):
        super().__init__()
        self.var_type = var_type
        self.src = src
        self.dest = dest
        self.X = X
    def _do(self, problem, n_samples, **kwargs):
        routes = population(n_samples, self.src, self.dest, self.X)
        #print(routes.shape)
        self.X = routes
        #print(self.X.shape)
        return self.X
    
class Crossover1(Crossover):
    def __init__(self, prob=0.7):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = crossOver(a,b)
        #print("Y:",Y)
        return Y
    
class Mutation1(Mutation):
    def __init__(self, prob =0.4):
        super().__init__()
        self.prob  = prob 

    def _do(self, problem, X, **kwargs):
        
        offsprings = np.zeros((2,1), dtype=object)
        # loop over individuals in population
        for idx,i in enumerate(X):
            # performe mutation with certain probability
            if np.random.uniform(0, 1) < self.prob:
                mutated_individual = mutate(i[0])
                #print("mutated_individual",mutated_individual, "###")
                offsprings[idx][0]=mutated_individual
        # if no mutation
            else:
                 offsprings[idx][0] =i[0]
        return offsprings
    
class RoutingProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=1, 
            n_obj=1, 
            n_constr=0)
    def _evaluate(self, X, out, *args, **kwargs):
        costs = route_cost(X)
        #print(costs.shape)
        out['F'] = np.column_stack([costs])


def optimize( strt, end, cost_mat,pop_size = 10, n_gen = 100, n_offspring = 4):
    #cost[nan_mask] = 20000000000* np.nanmax(cost) if np.nanmax(cost) else 0
    problem = RoutingProblem()
    algorithm = NSGA2(pop_size=pop_size,
                    sampling= Population(strt, end, cost_mat),
                    crossover= Crossover1(),
                    n_offsprings = 2,
                    mutation= Mutation1(),
                    eliminate_duplicates=False)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                algorithm,
                termination,
                save_history=True, 
                verbose=True)
    #stop = timeit.default_timer()
    #route_cost(res.X)
    return res