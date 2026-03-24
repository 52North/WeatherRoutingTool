import numpy as np
import pytest

from WeatherRoutingTool.algorithms.genetic.mcdm import RMethod


@pytest.mark.parametrize("obj_fuel,obj_time", [(1, 1), (1, 2), (2, 1)])
def test_weight_determination_for_solution_selection(plt, obj_fuel, obj_time):
    fuel_weight = np.random.rand(1, 10000) * 0.1
    time_weight = np.random.rand(1, 10000) * 0.1

    objective_dict = {
        "fuel_consumption": obj_fuel,
        "arrival_time": obj_time,
    }
    mcdm = RMethod(objective_dict)
    composite_weight = mcdm.get_composite_weight(
        sol_weight_list=[time_weight, fuel_weight],
        obj_weight_list=[obj_time, obj_fuel]
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scatter points
    ax.set_xlim(fuel_weight.max(), fuel_weight.min())
    ax.scatter(fuel_weight, time_weight, composite_weight, c=composite_weight, cmap='viridis', marker='o', s=40,
               alpha=0.6, edgecolors='w')

    # Set labels and title
    ax.set_title('3D Scatter Plot of Point Selections')
    ax.set_xlabel('fuel weight')
    ax.set_ylabel('time weight')
    ax.set_zlabel('composite weight')

    plt.saveas = f"test_composite_weight_fuel{obj_fuel}_time{obj_time}.png"


@pytest.mark.parametrize("rank,out", [
    (1, 1.),
    (2, 0.666666),
    (3, 0.545454),
    (4, 0.48),
])
def test_get_rank_sum(rank, out):
    objective_dict = {
        "fuel_consumption": 1,
        "arrival_time": 1,
    }
    mcdm = RMethod(objective_dict)
    res = mcdm.get_rank_sum(rank)
    assert np.isclose(res, out)


@pytest.mark.parametrize("rank,n_parts,out", [
    (4, 4, 0.48 / 2.69212),
    (50, 50, 0.2222614 / 15.287014),
    (1.5, 4, 0.309545),
])
def test_get_weigth_from_rank(rank, out, n_parts):
    objective_dict = {
        "fuel_consumption": 1,
        "arrival_time": 1,
    }
    rank_arr = np.array([rank])
    mcdm = RMethod(objective_dict)
    res = mcdm.get_weigths_from_rankarr(rank_arr, n_parts)
    assert np.isclose(res, out)
