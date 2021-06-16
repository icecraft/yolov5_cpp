from functools import partial
import jinja2

def expand_withname(name, args):
   if isinstance(args, jinja2.runtime.Undefined):
       return ''
   if isinstance(args, int):
       return '.{}({})'.format(name, args)
   return '.{}({})'.format(name, ', '.join(map(str, args)))


def just_expand(args):
   if isinstance(args, int):
       return '{}'.format(args)
   return  ', '.join(map(str, args))


def bool_(args):
    return 'true' if args else 'false'

def check_add(env, name, func):
    if name in env.filters:
        raise Exception('duplicate name filters')
    env.filters[name] = func


def register(env):    
    check_add(env, 'padding', partial(expand_withname, 'padding'))
    check_add(env, 'stride', partial(expand_withname, 'stride'))
    check_add(env, 'groups', partial(expand_withname, 'groups'))
    check_add(env, 'expand_arr_int', just_expand)
    check_add(env, 'bool', bool_)
