import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from models import rimless_wheel as model
from assignment_1_plot_RoA import (
    ATTRACTOR_COLORS,
    ROLLING,
    STANDING,
    classify_state_grid,
    draw_regions_of_attraction,
)
from assignment_1_plot_return_map import estimate_floquet_multiplier, find_fixed_point

SWEEP_SPOKE_COUNTS = (6, 7, 8, 9, 10, 11, 12)
SWEEP_SLOPES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def override_params(params, **overrides):
    # One parameter set with some entries replaced, leaving the original alone

    swept_params = dict(params)
    swept_params.update(overrides)

    return swept_params


def find_critical_slope(params, min_slope=1e-6, max_slope=0.6, n_refinements=60):
    # Shallowest slope with a rolling gait

    lower_slope, upper_slope = min_slope, max_slope

    for _ in range(n_refinements):
        middle_slope = 0.5 * (lower_slope + upper_slope)
        middle_params = override_params(params, slope=middle_slope)
        gait_margin = find_fixed_point(middle_params) - model.speed_to_clear_apex(
            middle_params
        )

        if gait_margin > 0:
            upper_slope = middle_slope
        else:
            lower_slope = middle_slope

    critical_slope = upper_slope

    return critical_slope


def measure_sweep(
    params,
    swept_key,
    swept_values,
    n_angles=201,
    n_velocities=201,
    max_initial_velocity=8.0,
):
    # How the basin and the local convergence rate move as one parameter varies.

    rolling_fractions = []
    floquet_multipliers = []

    for swept_value in swept_values:
        swept_params = override_params(params, **{swept_key: swept_value})

        _, _, label = classify_state_grid(
            swept_params, n_angles, n_velocities, max_initial_velocity
        )

        rolling_fractions.append(float((label == ROLLING).mean()))
        floquet_multipliers.append(float(estimate_floquet_multiplier(swept_params)))

    rolling_fractions = np.array(rolling_fractions)
    floquet_multipliers = np.array(floquet_multipliers)

    return rolling_fractions, floquet_multipliers


def plot_roa_matrix(params, slopes, spoke_counts, panel_resolution=201):
    # One region-of-attraction panel per (spokes, slope) pair
    # spokes increase to the right, inclination increases to the top.
    # Rows are walked in reverse because matplotlib counts row 0 from the top.

    n_rows, n_columns = len(slopes), len(spoke_counts)

    figure, axes_grid = plt.subplots(
        n_rows,
        n_columns,
        figsize=(1.5 * n_columns, 1.4 * n_rows),
        squeeze=False,
    )

    for row_index, slope in enumerate(reversed(slopes)):
        for column_index, n_spokes in enumerate(spoke_counts):
            axes = axes_grid[row_index, column_index]

            panel_params = override_params(params, slope=slope, n_spokes=n_spokes)

            label = draw_regions_of_attraction(
                axes, panel_params, panel_resolution, panel_resolution
            )
            rolling_percentage = float((label == ROLLING).mean()) * 100

            axes.set_xticks([])
            axes.set_yticks([])
            axes.text(
                0.5,
                0.04,
                f"{rolling_percentage:.0f}%",
                transform=axes.transAxes,
                ha="center",
                fontsize=7,
                color="white",
            )

            if row_index == 0:
                axes.set_title(f"N = {n_spokes}", fontsize=10)
            if column_index == 0:
                axes.set_ylabel(f"{slope:.2f}", fontsize=9, rotation=0, labelpad=16)

    figure.legend(
        handles=[
            Patch(facecolor=ATTRACTOR_COLORS(ROLLING), label="Rolls Forever"),
            Patch(facecolor=ATTRACTOR_COLORS(STANDING), label="Stops"),
        ],
        loc="lower left",
        framealpha=0.95,
        fontsize=9,
    )
    figure.suptitle(
        "Regions of attraction: spokes increasing right, slope (rad) increasing up"
    )
    figure.tight_layout()

    return figure


def plot_floquet_multiplier(params, slopes, spoke_counts):
    # Local convergence rate over both swept parameters in one axes. One series per
    # slope, and they land on identical values because the multiplier does not
    # depend on slope at all -- each simply starts at the shallowest spoke count
    # whose gait exists, since the rest come back nan.

    figure, axes = plt.subplots(figsize=(7.5, 5))

    markers = ("o", "s", "^", "v", "D", "P", "X", "*")

    for slope_index, slope in enumerate(slopes):
        multipliers = [
            estimate_floquet_multiplier(
                override_params(params, slope=slope, n_spokes=n_spokes)
            )
            for n_spokes in spoke_counts
        ]
        axes.plot(
            spoke_counts,
            multipliers,
            marker=markers[slope_index % len(markers)],
            ms=9,
            lw=1.2,
            alpha=0.55,
            label=f"slope = {slope:.2f}",
        )

    axes.set_xticks(spoke_counts)
    axes.set_ylim(0, 1.08)
    axes.set_xlabel("Number of Spokes")
    axes.set_ylabel("Floquet Multiplier")
    axes.set_title("Local convergence of the rolling gait")
    axes.legend(fontsize=8, loc="upper left", framealpha=0.95, ncol=2)

    figure.tight_layout()

    return figure


if __name__ == "__main__":
    params = model.generate_params()

    print(
        f"critical slope for N={params['n_spokes']}: "
        f"{find_critical_slope(params):.6f} rad"
    )

    for swept_key, swept_values in (
        ("slope", SWEEP_SLOPES),
        ("n_spokes", SWEEP_SPOKE_COUNTS),
    ):
        rolling_fractions, floquet_multipliers = measure_sweep(
            params, swept_key, swept_values
        )

        print(f"\nsweeping {swept_key}:")
        for swept_value, rolling_fraction, floquet_multiplier in zip(
            swept_values, rolling_fractions, floquet_multipliers
        ):
            fixed_point = find_fixed_point(
                override_params(params, **{swept_key: swept_value})
            )
            print(
                f"  {swept_value:>6g}: fixed point {fixed_point:6.3f} rad/s, "
                f"basin {rolling_fraction * 100:5.1f}%, "
                f"multiplier {floquet_multiplier:.4f}"
            )

    matrix_figure = plot_roa_matrix(params, SWEEP_SLOPES, SWEEP_SPOKE_COUNTS)
    multiplier_figure = plot_floquet_multiplier(
        params, SWEEP_SLOPES, SWEEP_SPOKE_COUNTS
    )

    plt.show()
