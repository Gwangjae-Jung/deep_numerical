def __getattr__(attr: str) -> object:
    if attr == "data_generators":
        from    .   import  data_generators
        return  data_generators
    if attr == "numerical":
        from    .   import  numerical
        return  numerical
    if attr == "pytorch":
        from    .   import  pytorch
        return  pytorch