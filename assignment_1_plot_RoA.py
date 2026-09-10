import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from models import rimless_wheel as model

ROLLING = 0
STANDING = 1
UNDECIDED = -1  # never survives a completed sweep; classify_states raises instead


def advance_to_next_impact(angle, angular_velocity, params):
    # Where does the state go next
    # Uses energy to avoid difficult integration

    gravity_over_length = params["gravity"] / params["length"]
    half_spoke_angle = np.pi / params["n_spokes"]
    trailing_angle = params["slope"] - half_spoke_angle
    leading_angle = params["slope"] + half_spoke_angle

    energy = 0.5 * angular_velocity**2 + gravity_over_length * np.cos(angle)

    heading_forward = (angular_velocity > 0) | ((angular_velocity == 0) & (angle > 0))

    climbing_apex = np.where(heading_forward, angle < 0, angle > 0)
    turns_back = climbing_apex & (energy <= gravity_over_length)
    leaves_forward = heading_forward ^ turns_back

    exit_angle = np.where(leaves_forward, leading_angle, trailing_angle)
    exit_speed = np.sqrt(
        np.maximum(2 * (energy - gravity_over_length * np.cos(exit_angle)), 0.0)
    )

    # Landing spoke becomes stance: the angle jumps to the opposite endpoint
    next_angle = np.where(leaves_forward, trailing_angle, leading_angle)
    next_velocity = np.where(leaves_forward, exit_speed, -exit_speed) * np.cos(2 * half_spoke_angle)

    settled = turns_back & heading_forward

    return next_angle, next_velocity, settled


def classify_states(angle, angular_velocity, params, rest_speed=1e-6, max_impacts=400):
    # Where does the state end up, either rolling or standing

    gravity_over_length = params["gravity"] / params["length"]
    half_spoke_angle = np.pi / params["n_spokes"]
    trailing_angle = params["slope"] - half_spoke_angle
    leading_angle = params["slope"] + half_spoke_angle
    speed_to_clear_apex = np.sqrt(2 * gravity_over_length * (1 - np.cos(trailing_angle)))

    # Clearing the apex once only guarantees clearing it forever when the rolling
    # limit cycle exists. The step-to-step map always converges to its fixed
    # point, so if that fixed point sits below the threshold the wheel steps down
    # past it and stops -- on those slopes there is no rolling attractor at all.
    momentum_scale = np.cos(2 * half_spoke_angle)
    swing_gain = 2 * gravity_over_length * (np.cos(trailing_angle) - np.cos(leading_angle))
    fixed_point = momentum_scale * np.sqrt(swing_gain / (1 - momentum_scale**2))
    rolling_exists = fixed_point > speed_to_clear_apex

    angle, angular_velocity = np.broadcast_arrays(angle, angular_velocity)
    angle = np.array(angle, dtype=float)
    angular_velocity = np.array(angular_velocity, dtype=float)

    label = np.full(angle.shape, UNDECIDED, dtype=int)

    # one impact per pass, applied only to the states still in play
    for _ in range(max_impacts):
        undecided = label == UNDECIDED
        if not undecided.any():
            break

        next_angle, next_velocity, settled = advance_to_next_impact(
            angle[undecided], angular_velocity[undecided], params
        )
        angle[undecided] = next_angle
        angular_velocity[undecided] = next_velocity

        stands = settled | (np.abs(next_velocity) < rest_speed)
        rolls = ~stands & rolling_exists & (next_velocity > speed_to_clear_apex)
        label[undecided] = np.where(stands, STANDING, np.where(rolls, ROLLING, UNDECIDED))

    return label


def plot_regions_of_attraction(params, n_angles=401, n_velocities=401, max_intial_velocity=8.0):
    # the stance interval is the full range of angles the wheel can occupy
    half_spoke_angle = np.pi / params["n_spokes"]
    trailing_angle = params["slope"] - half_spoke_angle
    leading_angle = params["slope"] + half_spoke_angle

    angles = np.linspace(trailing_angle, leading_angle, n_angles)
    velocities = np.linspace(-max_intial_velocity, max_intial_velocity, n_velocities)
    angle_grid, velocity_grid = np.meshgrid(angles, velocities)

    label = classify_states(angle_grid, velocity_grid, params)

    attractor_colors = ListedColormap(["#b2182b", "#2166ac"])

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.pcolormesh(
        angle_grid,
        velocity_grid,
        label,
        cmap=attractor_colors,
        norm=BoundaryNorm([-0.5, 0.5, 1.5], attractor_colors.N),
        shading="auto",
    )

    axes.set_xlabel("Initial position (rad)")
    axes.set_ylabel("Initial angular velocity (rad/s)")
    axes.set_title(
        f"Rimless wheel regions of attraction "
        f"(N={params['n_spokes']}, slope={params['slope']} rad)"
    )

    axes.legend(
        handles=[
            Patch(facecolor=attractor_colors(ROLLING), label="Rolls Forever"),
            Patch(facecolor=attractor_colors(STANDING), label="Stops"),
        ],
        loc="lower right",
        framealpha=0.95,
    )

    figure.tight_layout()

    return figure, label


if __name__ == "__main__":
    params = model.generate_params()
    half_spoke_angle = np.pi / params["n_spokes"]
    trailing_angle = params["slope"] - half_spoke_angle
    leading_angle = params["slope"] + half_spoke_angle
    gravity_over_length = params["gravity"] / params["length"]
    speed_to_clear_apex = np.sqrt(2 * gravity_over_length * (1 - np.cos(trailing_angle)))

    print(f"stance interval: ({trailing_angle:+.4f}, {leading_angle:+.4f}) rad")
    print(f"speed needed at trailing spoke to clear the apex: {speed_to_clear_apex:.6f} rad/s")

    figure, label = plot_regions_of_attraction(params)
    plt.show()
