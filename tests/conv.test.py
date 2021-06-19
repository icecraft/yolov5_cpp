import jinja2
import sys
import os
from utils import get_tools_dir

sys.path.append(os.path.dirname(get_tools_dir()))

from tools.filters import register
file_loader = jinja2.FileSystemLoader('{}/{}'.format(get_tools_dir(), 'templates'))

env = jinja2.Environment(loader=file_loader)
register(env)

def test_conv():
    template = env.get_template('Conv.new.tpl')
    out = template.render(input_channels=10, output_channels=20, kernel_size=[3, 4], padding=[1, 2], bias=False, stride=1)
    print(type(out))


if __name__ == '__main__':  
    test_conv()