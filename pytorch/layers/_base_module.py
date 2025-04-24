import  torch


##################################################
##################################################
__all__: list[str] = ['BaseModule']


##################################################
##################################################
class BaseModule(torch.nn.Module):
    """A custom base class for all deep learning architectures in this package.
    """
    def __init__(self):
        super().__init__()
        return
    
    
    def count_parameters(self) -> int:
        """Count the number of the learnable parameters in the module.
        Note that each complex parameter is counted as two parameters.
        """
        cnt = 0
        for p in self.parameters():
            c = 2 if p.is_complex() else 1
            cnt += c * p.numel()
        return cnt
    
    
    def __str__(self) -> str:
        msg: list[str] = []
        __half_line = '=' * 30
        _front = ''.join((__half_line, f'< {self.__class__.__name__} >', __half_line))
        _line  = '-' * len(_front)
        _back  = '=' * len(_front)
        
        msg.append(_front)
        msg.append(f"[ Subnetworks ]\n")
        for name, md in self.named_children():
            msg.append(f"* {name}")
            msg.append(str(md))
            msg.append('')
        
        msg.append(_line)
        msg.append(f"[ Parameters ]")
        named_params = self.named_parameters(recurse=False)
        for name, p in named_params:
            msg.append(f"( {name} )")
            msg.append(f"- Shape:       {list(p.shape)}")
            msg.append(f"- Data type:   {p.dtype}")
        
        msg.append(_line)
        msg.append(f"Number of parameters: {self.count_parameters()}")
        msg.append(_back)
        
        return '\n'.join(msg)


##################################################
##################################################
# End of file