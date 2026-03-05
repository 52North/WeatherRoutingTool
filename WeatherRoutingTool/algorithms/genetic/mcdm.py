import numpy as np
import pandas as pd


class MCDM:
    """Base Class for Multi-Criteria Decision Making (MCDM).

    This Class implements the base functionality for selecting a single solution from the set of non-dominated
    solutions found by the genetic algorithm.
    """

    def __init__(self, objectives: dict):
        self.n_objs = len(objectives)
        self.objectives = objectives

    def get_best_compromise(self, solutions):
        pass


class RMethod(MCDM):
    """
        Implements the R-Method for MCDM.

        The R-Method ranks alternatives by calculating a composite weight derived from
        rank-based objective importance and solution performance rankings. It is
        designed to identify the best compromise solution in multi-objective scenarios.

        If two objectives are supposed to have equal rank, the user needs to provide the mean between the two
        possible ranks, e.g. weight=1.5 for both objectives for optimisation with two objectives.

        References:
          R.V. Rao and R.J. Lakshmi, "Ranking of Pareto-optimal solutions and selecting the best solution in multi-and
          many-objective optimization problems using R-method". Soft Computing Letters 3 (2021) 100015

        :param objectives: dictionary of objective names (keys) and their weights (values).
        :type objectives: dict
    """

    def __init__(self, objectives: dict[str, int]):
        """
        Initialises the RMethod instance.

        :param objectives: dictionary of objective names (keys) and their weights (values).
        :type objectives: dict
        """
        super().__init__(objectives)

        self.objective_weights = {}

        for obj_str in self.objectives:
            self.objective_weights[obj_str] = self.get_weigths_from_rankarr(
                np.array([self.objectives[obj_str]]),
                self.n_objs
            )

    def rank_solutions(self, obj: np.ndarray, dec: bool = False) -> np.ndarray:
        """
        Ranks array content according to increasing (dec = False) or decreasing (dec = True) values.

        :param obj: Array to be ranked.
        :type obj: numpy.ndarray
        :param dec: If True, rank in descending order (highest value gets rank 1).
                    Defaults to False.
        :type dec: bool
        :return: array of ranks
        :rtype: numpy.ndarray
        """
        rank_ind = np.argsort(obj)
        if dec:
            rank_ind = rank_ind[::-1]
        rank = np.argsort(rank_ind)
        rank = rank + 1
        return rank

    def get_composite_weight(self, sol_weight_list: list[np.ndarray], obj_weight_list: list[float]) -> np.ndarray:
        """
        Calculate the composite weight for all non-dominated solutions based on solution weights and objective weights.

        Note: The current implementation is limited to problems with exactly two objectives.

        :param sol_weight_list: List of weights for each solution based on their performance wrt. each objective.
        :type sol_weight_list: list[numpy.ndarray]
        :param obj_weight_list: List of objective weights based on user ranking.
        :type obj_weight_list: list[float]
        :raises NotImplementedError: If the number of objectives is greater than two.
        :return: Calculated composite weights for all solutions.
        :rtype: numpy.ndarray
        """
        sign = [1, -1]
        denominator = 0
        summands = 0
        product = 1

        if len(sol_weight_list) > 2:
            raise NotImplementedError('Calculation of the composite weight of the R-method is not implemented for'
                                      'more than two objectives.')

        for i in range(len(sol_weight_list)):
            denominator = denominator + sign[i] * 1. / obj_weight_list[i] * sol_weight_list[i]
            product = product * sol_weight_list[i]
        denominator = np.abs(denominator) + 0.2

        for i in range(len(sol_weight_list)):
            summands = summands + sol_weight_list[i] / denominator * obj_weight_list[i] * obj_weight_list[i]

        composite_weight = product + summands
        return composite_weight

    def get_best_compromise(self, solutions: np.ndarray) -> int:
        """
        Find the index of the best compromise solution from a set of candidates.

        This method normalises the objective values for each solution, ranks the latter with respect to each objective,
        and calculates composite weights based on solution and objective weights.

        :param solutions: 2D array of objective values where rows are alternative solutions and columns are
          objective values.
        :type solutions: numpy.ndarray
        :return: The index of the optimal solution in the provided array.
        :rtype: int
        """
        debug = False
        sol_weight_list = []
        obj_weight_list = []

        if self.n_objs == 1:
            return solutions.argmin()

        if debug:
            print('solutions: ', solutions)
            print('solutions shape: ', solutions.shape)

        rmethod_table = pd.DataFrame()
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        i_obj = 0
        norm = 1.
        for obj_str in self.objectives:
            objective_values = solutions[:, i_obj]
            max_value = np.max(objective_values)
            if i_obj == 0:
                norm = max_value
            else:
                objective_values = objective_values * norm * 1. / max_value
            rmethod_table[obj_str + '_obj'] = objective_values
            rmethod_table[obj_str + '_rank'] = self.rank_solutions(objective_values)
            rmethod_table[obj_str + '_weight'] = self.get_weigths_from_rankarr(
                rmethod_table[obj_str + '_rank'].to_numpy(),
                len(solutions))
            i_obj += 1
            sol_weight_list.append(rmethod_table[obj_str + '_weight'].to_numpy())
            obj_weight_list.append(self.objective_weights[obj_str])

        if debug:
            print('rmethod table:', rmethod_table)

        rmethod_table['composite_weight'] = self.get_composite_weight(
            sol_weight_list=sol_weight_list,
            obj_weight_list=obj_weight_list,
        )
        rmethod_table['composite_rank'] = self.rank_solutions(rmethod_table['composite_weight'], True)
        best_ind = np.argmax(rmethod_table['composite_rank'].to_numpy())

        if debug:
            print('rmethod table:', rmethod_table)
            print('best index: ', rmethod_table.iloc[best_ind])
        return best_ind

    def get_rank_sum(self, rank_max: int) -> float:
        """
        Calculate the reciprocal of the harmonic sum for a given rank.

        :param rank_max: The rank to evaluate the sum for.
        :type rank_max: int
        :return: Reciprocal of the sum of (1/k) for k from 1 to rank_max.
        :rtype: float
        """
        rank_sum = 0
        for rk in range(1, rank_max + 1):
            rank_sum += 1 / rk
        return 1 / rank_sum

    def get_weight_from_rank(self, rank: int, n_parts: int) -> float:
        """
        Compute a normalized weight for an individual rank of a solutions or objective.

        :param rank: The specific rank of the item.
        :type rank: int
        :param n_parts: Total number of solutions/objectives in the set.
        :type n_parts: int
        :return: Normalized weight.
        :rtype: float
        """
        numerator = self.get_rank_sum(rank)
        denominator_sum = 0.

        for j in range(1, n_parts + 1):
            temp = self.get_rank_sum(j)
            denominator_sum += temp
        return numerator / denominator_sum

    def get_weigths_from_rankarr(self, rank_arr: np.ndarray, n_parts: int) -> np.ndarray:
        """
        Convert an array of ranks into an array of weights.

        Objectives or solutions can receive fractional ranks, if two of them are supposed to have equal ranks. These
        fractional ranks are handled by averaging the weights of the floor and ceiling integer ranks.

        :param rank_arr: Array of ranks.
        :type rank_arr: numpy.ndarray
        :param n_parts: Total number of solutions/objectives in the set.
        :type n_parts: int
        :return: Array containing the derived weights.
        :rtype: numpy.ndarray
        """
        weight_array = np.full(rank_arr.shape, -99.)

        for irank in range(0, rank_arr.shape[0]):
            if rank_arr[irank] % 1. != 0.:
                smaller = int(np.floor(rank_arr[irank]))
                larger = int(np.ceil(rank_arr[irank]))
                smaller_weight = self.get_weight_from_rank(smaller, n_parts)
                larger_weight = self.get_weight_from_rank(larger, n_parts)
                weight_array[irank] = (smaller_weight + larger_weight) / 2
            else:
                weight_array[irank] = self.get_weight_from_rank(int(rank_arr[irank]), n_parts)

        return weight_array
