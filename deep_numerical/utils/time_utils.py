from    typing      import  Callable, Iterable


__all__ = ['get_time_str', 'sec_to_hms', 'hms_to_sec', 'get_tqdm', 'get_trange']


##################################################
##################################################
def get_time_str(seconds: bool = True) -> str:
    """## Generates a formatted datetime string.

    ## Description
    Returns the current local datetime formatted as a string.

    ## Arguments
    `seconds` (`bool`, default: `True`): If `True`, includes seconds (`"%Y%m%d_%H%M%S"`). Otherwise, excludes seconds (`"%Y%m%d_%H%M"`).

    ## Returns
    `str`: Formatted datetime string.
    """
    from    datetime    import  datetime
    current_time = datetime.now()
    return current_time.strftime("%Y%m%d_%H%M%S" if seconds else "%Y%m%d_%H%M")


def sec_to_hms(seconds: float) -> str:
    """## Converts seconds to a HH:MM:SS string.

    ## Description
    Converts a duration given in seconds into a string formatted as hours, minutes, and seconds.

    ## Arguments
    `seconds` (`float`): Duration in seconds.

    ## Returns
    `str`: Formatted time delta string.
    """
    from    datetime    import  timedelta
    return str(timedelta(seconds=seconds))


def hms_to_sec(hours: int, minutes: int, seconds: int) -> int:
    """## Converts hours, minutes, and seconds to total seconds.

    ## Description
    Calculates the total number of seconds corresponding to the given hours, minutes, and seconds.

    ## Arguments
    `hours` (`int`): Hours component.
    `minutes` (`int`): Minutes component.
    `seconds` (`int`): Seconds component.

    ## Returns
    `int`: Total number of seconds.
    """
    from    datetime    import  timedelta
    return int(timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds())


##################################################
##################################################
def get_tqdm() -> Callable[[Iterable, object], Iterable]:
    """## Returns the appropriate tqdm progress bar function.

    ## Description
    Detects whether execution is running inside an interactive notebook (e.g., Jupyter)
    and returns `tqdm.notebook.tqdm` if available, or standard `tqdm.tqdm` otherwise.

    ## Arguments
    None.

    ## Returns
    `Callable[[Iterable, object], Iterable]`: The appropriate `tqdm` callable.
    """
    try:
        from    IPython     import  get_ipython
        if get_ipython() is not None:
            from    tqdm.notebook   import  tqdm
        else:
            from    tqdm            import  tqdm
    except Exception:
        from    tqdm    import  tqdm
    return  tqdm


def get_trange() -> Callable[[Iterable, object], Iterable]:
    """## Returns the appropriate trange progress bar function.

    ## Description
    Detects whether execution is running inside an interactive notebook (e.g., Jupyter)
    and returns `tqdm.notebook.trange` if available, or standard `tqdm.trange` otherwise.

    ## Arguments
    None.

    ## Returns
    `Callable[[Iterable, object], Iterable]`: The appropriate `trange` callable.
    """
    try:
        from    IPython     import  get_ipython
        if get_ipython() is not None:
            from    tqdm.notebook   import  trange
        else:
            from    tqdm            import  trange
    except Exception:
        from    tqdm    import  trange
    return  trange


##################################################
##################################################
# End of file