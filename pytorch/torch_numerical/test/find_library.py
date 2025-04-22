"""A script to append the path of the library."""
import  os, sys
from    platform    import  system


_new_path = os.getcwd()
_rfind_string: str
_system = system()
if _system == 'Windows':
    _rfind_string = '\\'
elif _system == 'Linux':
    _rfind_string = '/'
for _ in range(2):
    _new_path = _new_path[:_new_path.rfind(_rfind_string)]
    print(f"The following path will be appended.", f">>> [{_new_path}]", sep='\n')
sys.path.append(_new_path)


##################################################
##################################################
# End of file