import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from assignment_1_plot_RoA import advance_to_next_impact


def find_fixed_point(params):
    gravity_over_length = model.gravity_over_length(params)
    _, trailing_angle, leading_angle = model.stance_geometry(params)

    # Same algebra as classify_states, kept separate so each file stands alone --
    # change one and the other needs the same change

    # Speed loss from changing spokes
    momentum_scale = model.momentum_scale(params)
    # Speed gain from gravity
    swing_gain = 2 * gravity_over_length * (np.cos(trailing_angle) - np.cos(leading_angle))
    # Speed where gain from gravity = loss from impact
    fixed_point = momentum_scale * np.sqrt(swing_gain / (1 - momentum_scale**2))

    return fixed_point


def measure_return_map(angular_velocity, params):
    _, trailing_angle, _ = model.stance_geometry(params)
    speed_to_clear_apex = model.speed_to_clear_apex(params)

    angle = np.full(np.shape(angular_velocity), trailing_angle)
    _, next_velocity, _ = advance_to_next_impact(angle, angular_velocity, params)

    stays_on_section = np.asarray(angular_velocity) > speed_to_clear_apex
    next_velocity_on_section = np.where(stays_on_section, next_velocity, np.nan)

    return next_velocity_on_section


def estimate_floquet_multiplier(params, perturbation=1e-4):
    # Local slope of the return map at its fixed point
    # Comes back as nan when no rolling gait exists

    fixed_point = find_fixed_point(params)

    perturbed_velocities = np.array(
        [fixed_point - perturbation, fixed_point + perturbation]
    )
    next_velocity_below, next_velocity_above = measure_return_map(
        perturbed_velocities, params
    )

    floquet_multiplier = (next_velocity_above - next_velocity_below) / (2 * perturbation)

    return floquet_multiplier


def plot_return_map(params, max_velocity=3.0, n_velocities=600):
    speed_to_clear_apex = model.speed_to_clear_apex(params)
    fixed_point = find_fixed_point(params)
    rolling_exists = fixed_point > speed_to_clear_apex

    # Scale plot area
    if max_velocity > fixed_point:
        plot_limit = max_velocity
    else:
        plot_limit = 1.1 * fixed_point

    velocities = np.linspace(0.0, plot_limit, n_velocities)
    next_velocities = measure_return_map(velocities, params)

    figure, axes = plt.subplots(figsize=(7, 7))

    axes.plot([0, plot_limit], [0, plot_limit], "k--", lw=1.3, label="identity")
    axes.plot(velocities, next_velocities, color="#b2182b", lw=2.2, label="return map")

    if rolling_exists:
        axes.plot(
            [fixed_point],
            [fixed_point],
            "o",
            color="k",
            ms=8,
            zorder=5,
            label=f"fixed point = {fixed_point:.4f} rad/s",
        )

    axes.set_xlim(0, plot_limit)
    axes.set_ylim(0, plot_limit)
    axes.set_aspect("equal")
    axes.set_xlabel("Angular Velocity at Current Contact (rad/s)")
    axes.set_ylabel("Angular Velocity at Next contact (rad/s)")
    axes.set_title(
        f"Step-to-step return map  (N={params['n_spokes']}, "
        f"slope={params['slope']} rad)"
    )
    axes.legend(loc="lower right", framealpha=0.95)
    figure.tight_layout()

    return figure


if __name__ == "__main__":
    params = model.generate_params()
    fixed_point = find_fixed_point(params)

    print(f"fixed point of the step-to-step map: {fixed_point:.6f} rad/s")

    for perturbation in (1e-3, 1e-4, 1e-5, 1e-6):
        floquet_multiplier = estimate_floquet_multiplier(params, perturbation)
        print(
            f"floquet multiplier (perturbation={perturbation:.0e}): "
            f"{floquet_multiplier:.6f}"
        )

    figure = plot_return_map(params)
    plt.show()
