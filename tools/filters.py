from functools import partial
import jinja2


# suppose it is array
def torch_expanding_array(args):
    return str(args).replace('(', '{').replace(')', '}')

def cpp_vector_expand(args):
    return str(args).replace('[', '{').replace(']', '}')

def bool_(arg):
    if arg is None or arg is False:
        return 'false'
    return 'true'

def check_add(env, name, func):
    if name in env.filters:
        raise Exception('duplicate name filters')
    env.filters[name] = func
    
def register(env):    
    check_add(env, 'torch_expanding_array', torch_expanding_array)
    check_add(env, 'cpp_vector_expand', cpp_vector_expand)
    check_add(env, 'bool', bool_)
    

