import  warnings
def __getattr__(attr: str) -> object:
    if attr in ["numerical", "numerical_numpy"]:
        warnings.warn(
            "The submodule 'numerical' is not actively updated, and will be deprecated in the future. Instead, use 'pytorch.torch_numerical' to implement numerical methods.",
            DeprecationWarning
        )
        from    .   import  numerical_numpy
        return  numerical_numpy
    if attr == "pytorch":
        from    .   import  pytorch
        return  pytorch