import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (gravity / length) * np.sin(angle)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def detect_impact(t, state, params):
    slope = params["slope"]
    half_spoke_angle = np.pi / params["n_spokes"]

    angle = state[0]
    return abs(angle - slope) - half_spoke_angle


detect_impact.terminal = True
detect_impact.direction = 1  # only on crossings that leave the stance interval


def apply_reset(state, params):
    slope = params["slope"]
    half_spoke_angle = np.pi / params["n_spokes"]
    spoke_spacing = 2 * half_spoke_angle

    angle = state[0]
    angular_velocity = state[1]

    if angle >= slope + half_spoke_angle:  # rolling downhill, leading spoke lands
        angle = angle - spoke_spacing
        angular_velocity = angular_velocity * np.cos(spoke_spacing)
    elif angle <= slope - half_spoke_angle:  # rocking back, trailing spoke lands
        angle = angle + spoke_spacing
        angular_velocity = angular_velocity * np.cos(spoke_spacing)

    state = np.array([angle, angular_velocity])
    return state

def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "length": 1.0,  # spoke length, hub to tip (m)
        "mass": 1.0,  # point mass at the hub (kg)
        "n_spokes": 8,  # total spokes on the wheel (unitless)
        "slope": 0.2,  # angle of slop to ground (rad)
    }
    return params