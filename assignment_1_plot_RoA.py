import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from models import rimless_wheel as model

ROLLING = 0
STANDING = 1
UNDECIDED = -1  # still in play mid-sweep; survives only if max_impacts runs out

ATTRACTOR_COLORS = ListedColormap(["#b2182b", "#2166ac"])


def advance_to_next_impact(angle, angular_velocity, params):
    # Where does the state go next
    # Uses energy to avoid difficult integration

    gravity_over_length = model.gravity_over_length(params)
    _, trailing_angle, leading_angle = model.stance_geometry(params)
    momentum_scale = model.momentum_scale(params)

    energy = 0.5 * angular_velocity**2 + gravity_over_length * np.cos(angle)

    heading_forward = (angular_velocity > 0) | ((angular_velocity == 0) & (angle > 0))

    climbing_apex = np.where(heading_forward, angle < 0, angle > 0)
    turns_back = climbing_apex & (energy <= gravity_over_length)

    # It leaves through the leading spoke if it was heading forward, unless it
    # stalled short of the apex and reversed -- and a state heading backward that
    # stalls likewise reverses into leaving forward. Hence the exclusive or.
    leaves_forward = heading_forward ^ turns_back

    exit_angle = np.where(leaves_forward, leading_angle, trailing_angle)
    exit_speed = np.sqrt(
        np.maximum(2 * (energy - gravity_over_length * np.cos(exit_angle)), 0.0)
    )

    # Landing spoke becomes stance: the angle jumps to the opposite endpoint
    next_angle = np.where(leaves_forward, trailing_angle, leading_angle)
    next_velocity = np.where(leaves_forward, exit_speed, -exit_speed) * momentum_scale

    settled = turns_back & heading_forward

    return next_angle, next_velocity, settled


def classify_states(angle, angular_velocity, params, rest_speed=1e-6, max_impacts=400):
    # Where does the state end up, either rolling or standing

    gravity_over_length = model.gravity_over_length(params)
    _, trailing_angle, leading_angle = model.stance_geometry(params)
    speed_to_clear_apex = model.speed_to_clear_apex(params)

    # Speed loss from changing spokes
    momentum_scale = model.momentum_scale(params)
    # Speed gain from gravity
    swing_gain = 2 * gravity_over_length * (np.cos(trailing_angle) - np.cos(leading_angle))
    # Speed where gain from gravity = loss from impact
    fixed_point = momentum_scale * np.sqrt(swing_gain / (1 - momentum_scale**2))

    rolling_exists = fixed_point > speed_to_clear_apex

    # broadcast_arrays hands back read-only views, and the loop below assigns into
    # these, so the copies are required rather than tidy-up
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


def classify_state_grid(
    params, n_angles=401, n_velocities=401, max_initial_velocity=8.0
):
    # Classifies a grid covering the whole state space the wheel can start in, and
    # hands back the grids alongside the labels so callers can plot or measure them

    # the stance interval is the full range of angles the wheel can occupy
    _, trailing_angle, leading_angle = model.stance_geometry(params)

    angles = np.linspace(trailing_angle, leading_angle, n_angles)
    velocities = np.linspace(-max_initial_velocity, max_initial_velocity, n_velocities)
    angle_grid, velocity_grid = np.meshgrid(angles, velocities)

    label = classify_states(angle_grid, velocity_grid, params)

    return angle_grid, velocity_grid, label


def draw_regions_of_attraction(
    axes, params, n_angles=401, n_velocities=401, max_initial_velocity=8.0
):
    # Paints one region-of-attraction panel into an existing axes and hands back
    # the labels, so callers can measure the basin without classifying twice

    angle_grid, velocity_grid, label = classify_state_grid(
        params, n_angles, n_velocities, max_initial_velocity
    )

    axes.pcolormesh(
        angle_grid,
        velocity_grid,
        label,
        cmap=ATTRACTOR_COLORS,
        norm=BoundaryNorm([-0.5, 0.5, 1.5], ATTRACTOR_COLORS.N),
        shading="auto",
    )

    return label


def plot_regions_of_attraction(
    params, n_angles=401, n_velocities=401, max_initial_velocity=8.0
):
    figure, axes = plt.subplots(figsize=(8, 6))

    draw_regions_of_attraction(
        axes, params, n_angles, n_velocities, max_initial_velocity
    )

    axes.set_xlabel("Initial position (rad)")
    axes.set_ylabel("Initial angular velocity (rad/s)")
    axes.set_title(
        f"Rimless wheel regions of attraction "
        f"(N={params['n_spokes']}, slope={params['slope']} rad)"
    )

    axes.legend(
        handles=[
            Patch(facecolor=ATTRACTOR_COLORS(ROLLING), label="Rolls Forever"),
            Patch(facecolor=ATTRACTOR_COLORS(STANDING), label="Stops"),
        ],
        loc="lower right",
        framealpha=0.95,
    )

    figure.tight_layout()

    return figure


if __name__ == "__main__":
    params = model.generate_params()
    _, trailing_angle, leading_angle = model.stance_geometry(params)
    speed_to_clear_apex = model.speed_to_clear_apex(params)

    print(f"stance interval: ({trailing_angle:+.4f}, {leading_angle:+.4f}) rad")
    print(
        "speed needed at trailing spoke to clear the apex: "
        f"{speed_to_clear_apex:.6f} rad/s"
    )

    plot_regions_of_attraction(params)
    plt.show()
