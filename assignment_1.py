import matplotlib.pyplot as plt

from models import rimless_wheel as model
from assignment_1_plot_RoA import plot_regions_of_attraction
from assignment_1_plot_return_map import find_fixed_point, plot_return_map


if __name__ == "__main__":
    params = model.generate_params()

    fixed_point = find_fixed_point(params)

    print(f"N = {params['n_spokes']} spokes, slope = {params['slope']} rad")

    # regions of attraction plot
    figure, label = plot_regions_of_attraction(params)

    # step-to-step return map plot
    figure, fixed_point, rolling_exists = plot_return_map(params)

    plt.show()
