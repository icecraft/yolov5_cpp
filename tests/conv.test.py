import jinja2
import sys
import os
from misc import get_tools_dir
import yaml

sys.path.append(os.path.dirname(get_tools_dir()))

from tools.filters import register
file_loader = jinja2.FileSystemLoader('{}/{}'.format(get_tools_dir(), 'templates'))
from tools.layers_render import Model

env = jinja2.Environment(loader=file_loader)
register(env)

def test_conv():
    template = env.get_template('Conv.new.tpl')
    out = template.render(input_channels=10, output_channels=20, kernel_size=[3, 4], padding=[1, 2], bias=False, stride=1)


def test_model():
    sys.path.append('/Users/xurui/test/yolov5/venv/lib/python3.7/site-packages')
    import torch
    sys.path.append('/Users/xurui/test/yolov5')
    print(sys.path)
    from models.yolo import parse_model
    with open('/Users/xurui/test/yolov5/models/yolov5s.yaml') as f:
        data = yaml.safe_load(f)
    mm = parse_model(data, [3])
    
    with open('../tools/templates/workspace/models.cpp.tpl', 'w') as f:
        model = Model(mm[0])
        model.render(env, f)
     
if __name__ == '__main__':  
    # test_conv()
    test_model()