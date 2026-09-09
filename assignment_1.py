import numpy as np

def make_params(N=8, l=1.0, g=9.81, gamma=0.0):
    return dict(N=N, l=l, g=g, gamma=gamma, alpha=np.pi / N)

def continuous_dynamics(t, state, params):
    """Stance-phase pendulum dynamics: theta_ddot = (g/l) sin(theta)."""
    theta, thetadot = state
    g, l = params['g'], params['l']
    thetaddot = (g / l) * np.sin(theta)
    return np.array([thetadot, thetaddot])

def guard(t, state, params):
    """Zero-crossing: fires when theta reaches gamma + alpha (rolling forward)."""
    theta, thetadot = state
    gamma, alpha = params['gamma'], params['alpha']
    return theta - (gamma + alpha)

guard.terminal = True
guard.direction = 1   # only trigger on the increasing crossing (theta rising into it)

def reset(state, params):
    """Instantaneous switch to the next spoke: coordinate shift + momentum-conserving velocity update."""
    theta, thetadot = state
    alpha = params['alpha']
    theta_new = theta - 2 * alpha
    thetadot_new = thetadot * np.cos(2 * alpha)
    return np.array([theta_new, thetadot_new])