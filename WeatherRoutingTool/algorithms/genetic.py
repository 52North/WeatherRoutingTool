import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize


class Population(Sampling):
    def __init__(self, src, dest, util, var_type=np.float64):
        super().__init__()
        self.var_type = var_type
        self.src = src
        self.dest = dest
        self.util = util

    def _do(self, problem, n_samples, **kwargs):
        routes = self.util.population(n_samples, self.src, self.dest)
        # print(routes.shape)
        self.X = routes
        # print(self.X.shape)
        return self.X


class GeneticCrossover(Crossover):
    def __init__(self, util, prob=1):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob
        self.util = util

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.util.cross_over(a, b)
        # print("Y:",Y)
        return Y


class GeneticMutation(Mutation):
    def __init__(self, util, prob=0.4):
        super().__init__()
        self.prob = prob
        self.util = util

    def _do(self, problem, X, **kwargs):
        offsprings = np.zeros((len(X), 1), dtype=object)
        # loop over individuals in population
        for idx, i in enumerate(X):
            # perform mutation with certain probability
            if np.random.uniform(0, 1) < self.prob:
                mutated_individual = self.util.mutate(i[0])
                # print("mutated_individual", mutated_individual, "###")
                offsprings[idx][0] = mutated_individual
        # if no mutation
            else:
                offsprings[idx][0] = i[0]
        return offsprings


class RoutingProblem(Problem):
    """
    Class definition of the weather routing problem
    """
    def __init__(self, util):
        super().__init__(n_var=1, n_obj=1, n_constr=1)
        self.util = util

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Method defined by pymoo which has to be overriden
        :param x: numpy matrix with shape (rows: number of solutions/individuals, columns: number of design variables)
        :param out:
            out['F']: function values, vector of length of number of solutions
            out['G']: constraints
        :param args:
        :param kwargs:
        :return:
        """
        # costs = route_cost(X)
        costs = self.util.power_cost(x)
        constraints = self.util.route_const(x)
        # print(costs.shape)
        out['F'] = np.column_stack([costs])
        out['G'] = np.column_stack([constraints])


def optimize(strt, end, pop_size, n_gen, n_offspring, util):
    # cost[nan_mask] = 20000000000* np.nanmax(cost) if np.nanmax(cost) else 0
    problem = RoutingProblem(util)
    algorithm = NSGA2(pop_size=pop_size,
                      sampling=Population(strt, end, util),
                      crossover=GeneticCrossover(util),
                      n_offsprings=n_offspring,
                      mutation=GeneticMutation(util),
                      eliminate_duplicates=False,
                      return_least_infeasible=False)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=True,
                   verbose=True)
    # stop = timeit.default_timer()
    # route_cost(res.X)
    return res
