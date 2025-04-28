import  warnings
def __getattr__(attr: str) -> object:
    if attr == "numerical":
        warnings.warn(
            "The submodule 'numerical' is not actively updated, and will be deprecated in the future. Instead, use 'pytorch.torch_numerical' to implement numerical methods.",
            DeprecationWarning
        )
        from    .   import  numerical
        return  numerical
    if attr == "pytorch":
        from    .   import  pytorch
        return  pytorch