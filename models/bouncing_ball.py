import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    drag_coeff = params["drag_coeff"]

    height = state[0]
    linear_velocity = state[1]

    drag_force = 0.5 * drag_coeff * linear_velocity * abs(linear_velocity)
    linear_acceleration = -gravity - (drag_force / mass)

    state_derivative = np.array([linear_velocity, linear_acceleration])
    return state_derivative

def apply_reset(state, params):
    """Reverse and damp the velocity when the ball reaches the ground."""
    ground = params["ground"]
    cor = params["cor"]

    height = state[0]
    linear_velocity = state[1]

    if height <= ground:
        height = ground
        linear_velocity = -cor * linear_velocity

    state = np.array([height, linear_velocity])
    return state

def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "mass": 0.5,  # ball mass (as point mass) (kg)
        "cor": 0.8,  # Coefficient of restitution (unitless)
        "drag_coeff": 0.0,  # Drag Coefficient (kg*m^2/s)
        "ground": 0.0,  # Position of ground (m)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]
    ground = params.get("ground", 0.0)

    height = state[0]
    linear_velocity = state[1]

    kinetic_energy = 0.5 * mass * linear_velocity**2
    potential_energy = mass * gravity * (height - ground)
    return kinetic_energy, potential_energy