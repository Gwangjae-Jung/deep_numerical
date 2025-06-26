from    typing      import  Callable


##################################################
##################################################
__all__: list[str] = [
    'exponential_cosine',
]


##################################################
##################################################
# Learning rate scheduler
def exponential_cosine(
        period:     float,
        half_life:  float,
    ) -> Callable[[int], float]:
    from    math    import  cos, exp, log, pi
    assert period > 0
    assert half_life > 0
    omega   = 2*pi/period
    lambda_lr: Callable[[int], float] = \
        lambda epoch: 0.5 * (1+cos(omega*epoch)) * exp(-log(2) * epoch/half_life)
    return lambda_lr


##################################################
##################################################
# End of file