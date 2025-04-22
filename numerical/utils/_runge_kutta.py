from    typing          import  Callable
from    ..utils         import  ArrayData


##################################################
##################################################
__all__: list[str] = [
    # Order 2
    'one_step_RK2_Heun',
    'one_step_RK2_Ralston',
    # Order 3
    'one_step_RK3_Heun',
    'one_step_RK3_Ralston',
    # Order 4
    'one_step_RK4_classic',
]


##################################################
##################################################
# Runge-Kutta methods (one-step)
def one_step_RK2_Heun(
        t_curr:     float,
        y_curr:     ArrayData,
        delta_t:    float,
        derivative: Callable[[float, ArrayData], ArrayData],
    ) -> ArrayData:
    """
    Reference: The Butcher's tableaux can be found in [Wikipedia](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Heun's_method).
    """
    k1  = derivative(t_curr, y_curr)
    k2  = derivative(t_curr + delta_t, y_curr + delta_t*k1)
    return y_curr + delta_t * (k1 + k2) / 2


def one_step_RK2_Ralston(
        t_curr:     float,
        y_curr:     ArrayData,
        delta_t:    float,
        derivative: Callable[[float, ArrayData], ArrayData],
    ) -> ArrayData:
    """
    Reference: The Butcher's tableaux can be found in [Wikipedia](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Ralston's_method).
    """
    k1  = derivative(t_curr, y_curr)
    k2  = derivative(t_curr + (2/3)*delta_t, y_curr + (2/3)*delta_t*k1)
    return y_curr + delta_t * (k1 + 3*k2) / 4


def one_step_RK3_Heun(
        t_curr:     float,
        y_curr:     ArrayData,
        delta_t:    float,
        derivative: Callable[[float, ArrayData], ArrayData],
    ) -> ArrayData:
    """
    Reference: The Butcher's tableaux can be found in [Wikipedia](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Heun's_third-order_method).
    """
    k1  = derivative(t_curr, y_curr)
    k2  = derivative(t_curr + (1/3)*delta_t, y_curr + (1/3)*delta_t*k1)
    k3  = derivative(t_curr + (2/3)*delta_t, y_curr + (2/3)*delta_t*k2)
    return y_curr + delta_t * (k1 + 3*k3) / 4


def one_step_RK3_Ralston(
        t_curr:     float,
        y_curr:     ArrayData,
        delta_t:    float,
        derivative: Callable[[float, ArrayData], ArrayData],
    ) -> ArrayData:
    """
    Reference: The Butcher's tableaux can be found in [Wikipedia](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Ralston's_third-order_method).
    """
    k1  = derivative(t_curr, y_curr)
    k2  = derivative(t_curr + 0.50*delta_t, y_curr + 0.50*delta_t*k1)
    k3  = derivative(t_curr + 0.75*delta_t, y_curr + 0.75*delta_t*k2)
    return y_curr + delta_t * (2*k1 + 3*k2 + 4*k3) / 9


def one_step_RK4_classic(
        t_curr:     float,
        y_curr:     ArrayData,
        delta_t:    float,
        derivative: Callable[[float, ArrayData], ArrayData],
    ) -> ArrayData:
    """
    Reference: The Butcher's tableaux can be found in [Wikipedia](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method).
    """
    k1  = derivative(t_curr, y_curr)
    k2  = derivative(t_curr + 0.5*delta_t, y_curr + 0.5*delta_t*k1)
    k3  = derivative(t_curr + 0.5*delta_t, y_curr + 0.5*delta_t*k2)
    k4  = derivative(t_curr + delta_t, y_curr + delta_t*k3)
    return  y_curr + delta_t * (k1 + 2*k2 + 2*k3 + k4) / 6


##################################################
##################################################
# End of file