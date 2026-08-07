from    typing      import  Callable, Iterable


__all__ = ['get_time_str', 'sec_to_hms', 'hms_to_sec', 'get_tqdm', 'get_trange']


##################################################
##################################################
def get_time_str(seconds: bool = True) -> str:
    """Returns the datetime string of the form `{year}{month}{day}_{hour}{minute}{second}`."""
    from    datetime    import  datetime
    current_time = datetime.now()
    return current_time.strftime("%Y%m%d_%H%M%S" if seconds else "%Y%m%d_%H%M")
def sec_to_hms(seconds: float) -> str:
    from    datetime    import  timedelta
    return str(timedelta(seconds=seconds))
def hms_to_sec(hours: int, minutes: int, seconds: int) -> int:
    from    datetime    import  timedelta
    return int(timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds())


##################################################
##################################################
def get_tqdm() -> Callable[[Iterable, object], Iterable]:
    try:
        # Check if running inside a Jupyter Notebook
        from    IPython     import  get_ipython
        if 'IPython' in globals() and get_ipython() is not None:
            from    tqdm.notebook   import  tqdm
        else:
            from    tqdm            import  tqdm
    except:
        # If IPython not available (pure .py execution)
        from    tqdm    import  tqdm
    return  tqdm


def get_trange() -> Callable[[Iterable, object], Iterable]:
    try:
        # Check if running inside a Jupyter Notebook
        from    IPython     import  get_ipython
        if 'IPython' in globals() and get_ipython() is not None:
            from    tqdm.notebook   import  trange
        else:
            from    tqdm            import  trange
    except:
        # If IPython not available (pure .py execution)
        from    tqdm    import  trange
    return  trange


##################################################
##################################################
# End of file