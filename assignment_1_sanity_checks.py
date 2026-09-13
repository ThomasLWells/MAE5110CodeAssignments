import numpy as np

from models import rimless_wheel as model
from integrators import rk4
from integrators.rk4_events import find_impact_time, rk4_step
from assignment_1_plot_RoA import STANDING, advance_to_next_impact, classify_states
from assignment_1_plot_return_map import estimate_floquet_multiplier


def report(name, expected, measured, passed):
    # One line per check, expectation first so the comparison reads in order
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"        expected: {expected}")
    print(f"        measured: {measured}")

    return passed


def check_stance_energy_is_conserved(params, timestep=1e-5, swing_time=0.05):
    # The stance phase is an undriven inverted pendulum, so its energy must not
    # drift. This tests dynamics() alone, before any impact is involved.

    gravity_over_length = model.gravity_over_length(params)
    _, trailing_angle, _ = model.stance_geometry(params)

    # short enough that the wheel does not reach a spoke and trip the guard
    _, state_traj = rk4(np.array([trailing_angle, 1.5]), timestep, swing_time, model, params)

    energy = 0.5 * state_traj[1] ** 2 + gravity_over_length * np.cos(state_traj[0])
    energy_drift = float(energy.max() - energy.min())

    return report(
        "stance energy is conserved",
        "drift below 1e-10 over the swing",
        f"drift {energy_drift:.2e} about E = {energy[0]:.6f}",
        energy_drift < 1e-10,
    )


def check_impact_law(params):
    # A leading-spoke impact should hand the wheel to the next spoke: the angle
    # lands on the opposite end of the stance interval and the velocity is scaled
    # by exactly cos(2 * half_spoke_angle).

    _, trailing_angle, leading_angle = model.stance_geometry(params)
    momentum_scale = model.momentum_scale(params)

    pre_impact_velocity = 2.0
    post_angle, post_velocity = model.apply_reset(
        np.array([leading_angle, pre_impact_velocity]), params
    )

    angle_error = abs(post_angle - trailing_angle)
    velocity_error = abs(post_velocity - pre_impact_velocity * momentum_scale)

    return report(
        "impact law resets angle and scales velocity",
        f"angle -> {trailing_angle:+.6f} rad, velocity -> {pre_impact_velocity * momentum_scale:.6f} rad/s",
        f"angle -> {post_angle:+.6f} rad, velocity -> {post_velocity:.6f} rad/s",
        angle_error < 1e-12 and velocity_error < 1e-12,
    )


def check_guard_sign(params):
    # detect_impact is the event function, so its sign convention is what
    # terminal/direction rely on: negative inside the stance interval, zero at the
    # spokes, positive once a spoke has been passed.

    _, trailing_angle, leading_angle = model.stance_geometry(params)
    midpoint = 0.5 * (trailing_angle + leading_angle)

    inside = model.detect_impact(0.0, np.array([midpoint, 0.0]), params)
    at_trailing = model.detect_impact(0.0, np.array([trailing_angle, 0.0]), params)
    at_leading = model.detect_impact(0.0, np.array([leading_angle, 0.0]), params)
    outside = model.detect_impact(0.0, np.array([leading_angle + 0.1, 0.0]), params)

    return report(
        "impact guard changes sign at the spokes",
        "negative inside, zero at both spokes, positive outside",
        f"inside {inside:+.3e}, trailing {at_trailing:+.1e}, "
        f"leading {at_leading:+.1e}, outside {outside:+.3e}",
        inside < 0 and abs(at_trailing) < 1e-12 and abs(at_leading) < 1e-12 and outside > 0,
    )


def integrate_to_next_impact(angular_velocity, params, timestep):
    # Integrate the real dynamics and land exactly on the guard before resetting.
    # Bisecting with find_impact_time matters: sampling the trajectory at the next
    # grid point instead leaves a timestep-sized offset that looks like map error.

    _, trailing_angle, _ = model.stance_geometry(params)

    state = np.array([trailing_angle, angular_velocity])
    time = 0.0

    for _ in range(int(10.0 / timestep)):
        trial_state = rk4_step(time, state, timestep, model, params)

        if model.detect_impact(time + timestep, trial_state, params) >= 0:
            sub_step = find_impact_time(time, state, timestep, model, params)
            impact_state = rk4_step(time, state, sub_step, model, params)
            return model.apply_reset(impact_state, params)

        state = trial_state
        time = time + timestep

    return None


def check_energy_map_matches_integration(params):
    # advance_to_next_impact skips integration entirely and uses conserved energy,
    # which is why a 401x401 region-of-attraction grid runs in milliseconds. That
    # shortcut is only legitimate if it reproduces the integrated trajectory.

    _, trailing_angle, _ = model.stance_geometry(params)

    worst_error = 0.0
    samples = []

    for angular_velocity in (1.2, 1.5, 2.5, 4.0):
        _, mapped_velocity, _ = advance_to_next_impact(
            np.array(trailing_angle), np.array(angular_velocity), params
        )

        for timestep in (1e-3, 1e-4):
            integrated = integrate_to_next_impact(angular_velocity, params, timestep)
            error = abs(float(mapped_velocity) - integrated[1])
            worst_error = max(worst_error, error)

        samples.append(f"w0={angular_velocity:.1f} -> {float(mapped_velocity):.8f}")

    return report(
        "energy map matches full integration with event bisection",
        "post-impact velocities agree to better than 1e-9",
        f"worst error {worst_error:.2e} over {', '.join(samples)}",
        worst_error < 1e-9,
    )


def check_flat_ground_never_rolls(params, n_angles=201, n_velocities=201):
    # With no slope there is no energy input, while every impact still dissipates,
    # so no rolling gait can exist no matter how fast the wheel starts.

    flat_params = dict(params)
    flat_params["slope"] = 0.0

    gravity_over_length = model.gravity_over_length(flat_params)
    _, trailing_angle, leading_angle = model.stance_geometry(flat_params)
    swing_gain = (
        2 * gravity_over_length * (np.cos(trailing_angle) - np.cos(leading_angle))
    )

    angles = np.linspace(trailing_angle, leading_angle, n_angles)
    velocities = np.linspace(-8.0, 8.0, n_velocities)
    angle_grid, velocity_grid = np.meshgrid(angles, velocities)
    label = classify_states(angle_grid, velocity_grid, flat_params)

    all_standing = bool((label == STANDING).all())

    return report(
        "flat ground has no rolling gait",
        "swing_gain exactly 0 and every initial state stops",
        f"swing_gain {swing_gain:.1e}, "
        f"{(label == STANDING).mean() * 100:.1f}% of states stop",
        swing_gain == 0.0 and all_standing,
    )


def check_floquet_multiplier_closed_form(params, spoke_counts=(6, 7, 8, 9, 10, 11, 12)):
    # Differentiating the return map gives f'(w) = c*w/sqrt(w**2 + G), and at the
    # fixed point sqrt(w**2 + G) = w/c, so the multiplier collapses to c**2 with no
    # dependence on slope. The sweep figure rests on that, so check it directly.

    worst_error = 0.0

    for n_spokes in spoke_counts:
        for slope in (0.20, 0.30, 0.40):
            swept_params = dict(params)
            swept_params["n_spokes"] = n_spokes
            swept_params["slope"] = slope

            estimated = estimate_floquet_multiplier(swept_params)
            closed_form = model.momentum_scale(swept_params) ** 2

            if np.isfinite(estimated):
                worst_error = max(worst_error, abs(estimated - closed_form))

    return report(
        "Floquet multiplier equals cos^2(2 * half_spoke_angle)",
        "agreement better than 1e-6, independent of slope",
        f"worst error {worst_error:.2e} over N={spoke_counts[0]}..{spoke_counts[-1]} "
        "at slopes 0.20/0.30/0.40",
        worst_error < 1e-6,
    )


if __name__ == "__main__":
    params = model.generate_params()

    print(f"rimless wheel sanity checks (N={params['n_spokes']}, slope={params['slope']} rad)\n")

    results = [
        check_stance_energy_is_conserved(params),
        check_impact_law(params),
        check_guard_sign(params),
        check_energy_map_matches_integration(params),
        check_flat_ground_never_rolls(params),
        check_floquet_multiplier_closed_form(params),
    ]

    print(f"\n{sum(results)}/{len(results)} checks passed")
