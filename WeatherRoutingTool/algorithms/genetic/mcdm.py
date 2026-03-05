import numpy as np
import pandas as pd


class MCDM:
    """Base Class for Multi-Criteria Decision Making

    This Class implements the base functionality for selecting a single solution from the set of non-dominated
    solutions found by the genetic algorithm.
    """

    def __init__(self, objectives: dict):
        self.n_objs = len(objectives)
        self.objectives = objectives

    def get_best_compromise(self, solutions):
        pass


class RMethod(MCDM):
    def __init__(self, objectives: dict):
        super().__init__(objectives)

        self.objective_weights = {}

        for obj_str in self.objectives:
            self.objective_weights[obj_str] = self.get_weigths_from_rankarr(
                np.array([self.objectives[obj_str]]),
                self.n_objs
            )
        print('n_objs: ', self.n_objs)
        print('objectives: ', self.objectives)

    def rank_solutions(self, obj, dec=False):
        rank_ind = np.argsort(obj)
        if dec:
            rank_ind = rank_ind[::-1]
        rank = np.argsort(rank_ind)
        rank = rank + 1
        return rank

    def get_composite_weight(self, sol_weight_list, obj_weight_list):
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

    def get_best_compromise(self, solutions):
        debug = True
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

    def get_rank_sum(self, rank_max):
        rank_sum = 0
        for rk in range(1, rank_max + 1):
            rank_sum += 1 / rk
        return 1 / rank_sum

    def get_weight_from_rank(self, rank, n_parts):
        numerator = self.get_rank_sum(rank)
        denominator_sum = 0.

        for j in range(1, n_parts + 1):
            temp = self.get_rank_sum(j)
            denominator_sum += temp
        return numerator / denominator_sum

    def get_weigths_from_rankarr(self, rank_arr, n_parts):
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
