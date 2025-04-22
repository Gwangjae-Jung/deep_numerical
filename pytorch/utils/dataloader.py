from    typing                  import  Union
from    typing_extensions       import  Self
from    platform                import  system
from    pathlib                 import  Path
from    numpy                   import  ndarray, load
from    numpy.lib.npyio         import  NpzFile
from    ...utils_main._dtype    import  PathLike



##################################################
##################################################
__all__ = [
    "PDE_PATH__BURGERS",
    "PDE_PATH__DARCY_FLOW",
    "npzReader",
]


##################################################
##################################################
_sys = system()
_root_list = {
    'Windows':  Path("E:/PDE_datasets"),
    'Linux':    Path("/ssd/PANGPANG"),
}
PDE_PATH__BURGERS: dict[str, Path] = {
    'R10':          Path("from_pde_dataset/Burgers equation/Burgers_R10.npz"),
    'v100_t100':    Path("from_pde_dataset/Burgers equation/Burgers_v100_t100.npz"),
    'v1000_t200':   Path("from_pde_dataset/Burgers equation/Burgers_v1000_t200.npz"),
}
PDE_PATH__DARCY_FLOW: dict[str, Path] = {
    '241_1':        Path("from_pde_dataset/Darcy flow/Darcy241_1.npz"),
    '241_2':        Path("from_pde_dataset/Darcy flow/Darcy241_2.npz"),
    '421_1':        Path("from_pde_dataset/Darcy flow/Darcy421_1.npz"),
    '421_2':        Path("from_pde_dataset/Darcy flow/Darcy421_2.npz"),
}
for k in PDE_PATH__BURGERS.keys():
    PDE_PATH__BURGERS[k] = _root_list[_sys] / PDE_PATH__BURGERS[k]
for k in PDE_PATH__DARCY_FLOW.keys():
    PDE_PATH__DARCY_FLOW[k] = _root_list[_sys] / PDE_PATH__DARCY_FLOW[k]
    
    
##################################################
##################################################
class npzReader():
    """## `npz` reader
    ### A class to read `.npz` files
    
    -----
    ### Description
    By passing the path of an `npz` file to be loaded, this class generates an object which is ready to load data from the file.
    One can also close the file or change the path of the file to be loaded.
    """
    def __init__(
            self,
            path:   PathLike,
        ) -> Self:
        self.npz: NpzFile   = load(path, allow_pickle = True)
        return

    
    def get_field(self, key: str) -> Union[ndarray, object]:
        if key in self.npz.keys():
            return self.npz[key]
        else:
            print(f"Cannot find '{key}' from the following file:")
            print("*\t"f"{self.path}")
            return None
    
    
    def close_file(self) -> None:
        self.npz.close()
        return
    
    
    def reset(
            self,
            path:   PathLike,
        ) -> None:
        self.close_file()
        self.__init__(path)
        return
    
    
    def __repr__(self) -> str:
        msg: list[str] = []
        msg.append('='*20 + '< npzReader >' + '='*20)
        for k in self.keys:
            v = self.get_field(k)
            if not hasattr(v, "ndim") or v.ndim==0:
                msg.append(f"* {k:30s}: {v}")
            else:
                msg.append(f"* {k+'.shape':30s}: {v.shape}")
        msg.append( '='*len(msg[0]) )
        return '\n'.join(msg)
    

    @property
    def is_open(self) -> bool:
        if self.npz.fid != None:
            return True
        else:
            return False
    @property
    def path(self) -> str:
        if self.is_open:
            return self.npz.fid.name
        else:
            print(f"Currently, the npz reader is closed.")
            return None
    @property
    def keys(self) -> tuple:
        if self.is_open:
            return tuple([k for k in self.npz.keys()])
        else:
            print(f"Currently, the npz reader is closed.")
            return None


##################################################
##################################################
# End of file