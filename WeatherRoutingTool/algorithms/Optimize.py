from pymoo.factory import get_termination
from pymoo.optimize import minimize

pop_size = 30
n_gen = 50
n_offspring = 4
# cost[nan_mask] = 20000000000* np.nanmax(cost) if np.nanmax(cost) else 0
problem = RoutingProblem()
algorithm = NSGA2(pop_size=pop_size,
                  sampling=Population(start, end, cost),
                  crossover=GeneticCrossover(),
                  n_offsprings=n_offspring,
                  mutation=GeneticMutation(),
                  eliminate_duplicates=False)
termination = get_termination("n_gen", n_gen)

res = minimize(problem,
               algorithm,
               termination,
               save_history=True, 
               verbose=True)
# stop = timeit.default_timer()
# route_cost(res.X)
