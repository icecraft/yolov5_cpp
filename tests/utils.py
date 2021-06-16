import os

def get_tools_dir():
    
    return os.path.dirname(os.path.realpath(__file__)).replace('tests', 'tools')