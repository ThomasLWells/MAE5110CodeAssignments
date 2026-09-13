import matplotlib.pyplot as plt

from models import rimless_wheel as model
from assignment_1_plot_RoA import plot_regions_of_attraction
from assignment_1_plot_return_map import (
    estimate_floquet_multiplier,
    find_fixed_point,
    plot_return_map,
)
from assignment_1_plot_sweeps import (
    SWEEP_SLOPES,
    SWEEP_SPOKE_COUNTS,
    plot_floquet_multiplier,
    plot_roa_matrix,
)


if __name__ == "__main__":
    params = model.generate_params()

    fixed_point = find_fixed_point(params)
    floquet_multiplier = estimate_floquet_multiplier(params)

    print(f"N = {params['n_spokes']} spokes, slope = {params['slope']} rad")
    print(f"fixed point of the step-to-step map: {fixed_point:.6f} rad/s")
    print(f"floquet multiplier of the rolling gait: {floquet_multiplier:.6f}")

    # regions of attraction plot
    plot_regions_of_attraction(params)

    # step-to-step return map plot
    plot_return_map(params)

    # how slope and spoke count move the basin and the convergence rate
    plot_roa_matrix(params, SWEEP_SLOPES, SWEEP_SPOKE_COUNTS)
    plot_floquet_multiplier(params, SWEEP_SLOPES, SWEEP_SPOKE_COUNTS)

    plt.show()
