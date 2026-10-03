from   typing         import Callable
from   deep_numerical import ArrayData


__all__: list[str] = [
    # Order 1
    'RK1_Euler',
    'one_step_RK1_Euler',
    # Order 2
    'RK2_Heun',
    'RK2_Ralston',
    'one_step_RK2_Heun',
    'one_step_RK2_Ralston',
    # Order 3
    'RK3_Heun',
    'RK3_Ralston',
    'one_step_RK3_Heun',
    'one_step_RK3_Ralston',
    # Order 4
    'RK4_classic',
    'one_step_RK4_classic',
]


##################################################
def RK1_Euler(
    t_curr:     float,
    y_curr:     ArrayData,
    delta_t:    float,
    derivative: Callable[[float, ArrayData], ArrayData],
) -> ArrayData:
    """First-order forward Euler one-step numerical integration.

    ## Description
    Computes a single time-step advancement using the first-order forward Euler method.

    ## Arguments
    `t_curr` (`float`): Current simulation time.
    `y_curr` (`ArrayData`): Current state tensor or array.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `derivative` (`Callable[[float, ArrayData], ArrayData]`): Function evaluating $dy/dt = f(t, y)$.

    ## Returns
    `ArrayData`: Advanced state $y(t + \\Delta t)$.
    """
    k1: ArrayData = derivative(t_curr, y_curr)
    return y_curr + delta_t * k1


def RK2_Heun(
    t_curr:     float,
    y_curr:     ArrayData,
    delta_t:    float,
    derivative: Callable[[float, ArrayData], ArrayData],
) -> ArrayData:
    """Second-order Heun one-step numerical integration.

    ## Description
    Computes a single time-step advancement using Heun's second-order Runge-Kutta method.

    ## Arguments
    `t_curr` (`float`): Current simulation time.
    `y_curr` (`ArrayData`): Current state tensor or array.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `derivative` (`Callable[[float, ArrayData], ArrayData]`): Function evaluating $dy/dt = f(t, y)$.

    ## Returns
    `ArrayData`: Advanced state $y(t + \\Delta t)$.
    """
    k1: ArrayData = derivative(t_curr, y_curr)
    k2: ArrayData = derivative(t_curr + delta_t, y_curr + delta_t * k1)
    return y_curr + delta_t * (k1 + k2) / 2


def RK2_Ralston(
    t_curr:     float,
    y_curr:     ArrayData,
    delta_t:    float,
    derivative: Callable[[float, ArrayData], ArrayData],
) -> ArrayData:
    """Second-order Ralston one-step numerical integration.

    ## Description
    Computes a single time-step advancement using Ralston's second-order Runge-Kutta method with minimum truncation error.

    ## Arguments
    `t_curr` (`float`): Current simulation time.
    `y_curr` (`ArrayData`): Current state tensor or array.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `derivative` (`Callable[[float, ArrayData], ArrayData]`): Function evaluating $dy/dt = f(t, y)$.

    ## Returns
    `ArrayData`: Advanced state $y(t + \\Delta t)$.
    """
    k1: ArrayData = derivative(t_curr, y_curr)
    k2: ArrayData = derivative(t_curr + (2 / 3) * delta_t, y_curr + (2 / 3) * delta_t * k1)
    return y_curr + delta_t * (k1 + 3 * k2) / 4


def RK3_Heun(
    t_curr:     float,
    y_curr:     ArrayData,
    delta_t:    float,
    derivative: Callable[[float, ArrayData], ArrayData],
) -> ArrayData:
    """Third-order Heun one-step numerical integration.

    ## Description
    Computes a single time-step advancement using Heun's third-order Runge-Kutta method.

    ## Arguments
    `t_curr` (`float`): Current simulation time.
    `y_curr` (`ArrayData`): Current state tensor or array.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `derivative` (`Callable[[float, ArrayData], ArrayData]`): Function evaluating $dy/dt = f(t, y)$.

    ## Returns
    `ArrayData`: Advanced state $y(t + \\Delta t)$.
    """
    k1: ArrayData = derivative(t_curr, y_curr)
    k2: ArrayData = derivative(t_curr + (1 / 3) * delta_t, y_curr + (1 / 3) * delta_t * k1)
    k3: ArrayData = derivative(t_curr + (2 / 3) * delta_t, y_curr + (2 / 3) * delta_t * k2)
    return y_curr + delta_t * (k1 + 3 * k3) / 4


def RK3_Ralston(
    t_curr:     float,
    y_curr:     ArrayData,
    delta_t:    float,
    derivative: Callable[[float, ArrayData], ArrayData],
) -> ArrayData:
    """Third-order Ralston one-step numerical integration.

    ## Description
    Computes a single time-step advancement using Ralston's third-order Runge-Kutta method.

    ## Arguments
    `t_curr` (`float`): Current simulation time.
    `y_curr` (`ArrayData`): Current state tensor or array.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `derivative` (`Callable[[float, ArrayData], ArrayData]`): Function evaluating $dy/dt = f(t, y)$.

    ## Returns
    `ArrayData`: Advanced state $y(t + \\Delta t)$.
    """
    k1: ArrayData = derivative(t_curr, y_curr)
    k2: ArrayData = derivative(t_curr + 0.50 * delta_t, y_curr + 0.50 * delta_t * k1)
    k3: ArrayData = derivative(t_curr + 0.75 * delta_t, y_curr + 0.75 * delta_t * k2)
    return y_curr + delta_t * (2 * k1 + 3 * k2 + 4 * k3) / 9


def RK4_classic(
    t_curr:     float,
    y_curr:     ArrayData,
    delta_t:    float,
    derivative: Callable[[float, ArrayData], ArrayData],
) -> ArrayData:
    """Classic fourth-order Runge-Kutta one-step numerical integration.

    ## Description
    Computes a single time-step advancement using the classical fourth-order Runge-Kutta (RK4) method.

    ## Arguments
    `t_curr` (`float`): Current simulation time.
    `y_curr` (`ArrayData`): Current state tensor or array.
    `delta_t` (`float`): Time step size $\\Delta t$.
    `derivative` (`Callable[[float, ArrayData], ArrayData]`): Function evaluating $dy/dt = f(t, y)$.

    ## Returns
    `ArrayData`: Advanced state $y(t + \\Delta t)$.
    """
    k1: ArrayData = derivative(t_curr, y_curr)
    k2: ArrayData = derivative(t_curr + 0.5 * delta_t, y_curr + 0.5 * delta_t * k1)
    k3: ArrayData = derivative(t_curr + 0.5 * delta_t, y_curr + 0.5 * delta_t * k2)
    k4: ArrayData = derivative(t_curr + delta_t, y_curr + delta_t * k3)
    return y_curr + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6


one_step_RK1_Euler:   Callable[..., ArrayData] = RK1_Euler
one_step_RK2_Heun:    Callable[..., ArrayData] = RK2_Heun
one_step_RK2_Ralston: Callable[..., ArrayData] = RK2_Ralston
one_step_RK3_Heun:    Callable[..., ArrayData] = RK3_Heun
one_step_RK3_Ralston: Callable[..., ArrayData] = RK3_Ralston
one_step_RK4_classic: Callable[..., ArrayData] = RK4_classic


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()