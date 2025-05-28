import  warnings
def __getattr__(attr: str) -> object:
    if attr == "pytorch":
        from    .   import  deep_numerical
        return  deep_numerical