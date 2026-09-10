import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from assignment_1_plot_RoA import advance_to_next_impact


def find_fixed_point(params):
    half_spoke_angle = np.pi / params["n_spokes"]
    trailing_angle = params["slope"] - half_spoke_angle
    leading_angle = params["slope"] + half_spoke_angle
    gravity_over_length = params["gravity"] / params["length"]
    momentum_scale = np.cos(2 * half_spoke_angle)

    swing_gain = 2 * gravity_over_length * (np.cos(trailing_angle) - np.cos(leading_angle))
    return momentum_scale * np.sqrt(swing_gain / (1 - momentum_scale**2))


def measure_return_map(post_impact_velocity, params):
    gravity_over_length = params["gravity"] / params["length"]
    trailing_angle = params["slope"] - np.pi / params["n_spokes"]
    speed_to_clear_apex = np.sqrt(2 * gravity_over_length * (1 - np.cos(trailing_angle)))

    angle = np.full(np.shape(post_impact_velocity), trailing_angle)
    _, next_velocity, _ = advance_to_next_impact(angle, post_impact_velocity, params)

    stays_on_section = np.asarray(post_impact_velocity) > speed_to_clear_apex
    return np.where(stays_on_section, next_velocity, np.nan)


def plot_return_map(params, max_velocity=3.0, n_samples=600):
    gravity_over_length = params["gravity"] / params["length"]
    trailing_angle = params["slope"] - np.pi / params["n_spokes"]
    speed_to_clear_apex = np.sqrt(2 * gravity_over_length * (1 - np.cos(trailing_angle)))
    fixed_point = find_fixed_point(params)
    rolling_exists = fixed_point > speed_to_clear_apex

    velocities = np.linspace(0.0, max_velocity, n_samples)
    next_velocities = measure_return_map(velocities, params)

    figure, axes = plt.subplots(figsize=(7, 7))

    axes.plot([0, max_velocity], [0, max_velocity], "k--", lw=1.3, label="identity")
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

    axes.set_xlim(0, max_velocity)
    axes.set_ylim(0, max_velocity)
    axes.set_aspect("equal")
    axes.set_xlabel(r"$\omega^+_n$  at one contact (rad/s)")
    axes.set_ylabel(r"$\omega^+_{n+1}$  at the next contact (rad/s)")
    axes.set_title(
        f"Step-to-step return map  (N={params['n_spokes']}, "
        f"slope={params['slope']} rad)"
    )
    axes.legend(loc="lower right", framealpha=0.95)
    figure.tight_layout()

    return figure, fixed_point, rolling_exists


if __name__ == "__main__":
    params = model.generate_params()
    gravity_over_length = params["gravity"] / params["length"]
    trailing_angle = params["slope"] - np.pi / params["n_spokes"]
    speed_to_clear_apex = np.sqrt(2 * gravity_over_length * (1 - np.cos(trailing_angle)))
    fixed_point = find_fixed_point(params)

    figure, fixed_point, rolling_exists = plot_return_map(params)
    plt.show()
