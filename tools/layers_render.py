
import os
import shutil

def prepare_env():
    fn = os.path.dirname(__file__) + "/templates/workspace"
    if os.path.exists(fn):
        shutil.rmtree(fn)
    os.mkdir(fn)

def get_torch_layer_name(layer):
    return "nn." + layer.type.split(".")[-1]


def is_yolo_layer(layer):
    full_name = layer.type
    if full_name.startswith("torch."):
        return False, get_torch_layer_name(layer)
    return True, full_name.split(".")[-1]


class Registry(type):
    entries = {}
    def __init__(cls, name, bases, attrs):
            Registry.entries[name[1:]] = cls


class Rbase(metaclass=Registry):
    count = 0
    
    @classmethod
    def get_seq(cls):
        cls.count += 1
        return cls.count
    
    def get_tpl_name(self):
        return "{}.new.tpl".format(self.__class__.__name__[1:])
        
    def render(self, env, fobj):
        print(self.__class__.__name__)
        
        template = env.get_template(self.get_tpl_name())
        data = self._render()
        ret = template.render(**data)
        fobj.write(ret)
        fobj.flush()
        

class RFocus(Rbase):
    
    def __init__(self, layer):
        self.focus = layer
        
    def _render(self):
        conv = self.focus.conv
        d = {}
        d['in_channels'] = conv.conv.in_channels
        d['out_channels'] = conv.conv.out_channels
        d['kernel_size'] = conv.conv.kernel_size
        d['stride'] = conv.conv.stride
        d['padding'] = conv.conv.padding
        d['bias'] = conv.conv.bias 
        d['dilation'] = conv.conv.dilation
        d['groups'] = conv.conv.groups
        d['seq'] = RFocus.get_seq()
        return d
        
      
class RConv(Rbase):
    def __init__(self, layer):
        self.conv = layer
        
    def _render(self):
        conv = self.conv 
        
        d = {}
        d['in_channels'] = conv.conv.in_channels
        d['out_channels'] = conv.conv.out_channels
        d['kernel_size'] = conv.conv.kernel_size
        d['stride'] = conv.conv.stride
        d['padding'] = conv.conv.padding
        d['bias'] = conv.conv.bias
        d['dilation'] = conv.conv.dilation
        d['groups'] = conv.conv.groups
        d['seq'] = RConv.get_seq()
        return d


class RC3(Rbase):
    def __init__(self, layer):
        self.c3 = layer
        
    def _render(self):
        conv = self.c3.cv1
        
        d = {}
        d['in_channels'] = conv.conv.in_channels
        d['out_channels'] = conv.conv.out_channels
        d['kernel_size'] = conv.conv.kernel_size
        d['stride'] = conv.conv.stride
        d['padding'] = conv.conv.padding
        d['bias'] = conv.conv.bias
        d['dilation'] = conv.conv.dilation
        d['groups'] = conv.conv.groups
    
        d['n'] = len(self.c3.m)
        d['eps'] = d['out_channels']/d['in_channels']
        d['shortcut'] = self.c3.m[0].add
        d['groups2'] =  self.c3.m[0].cv2.conv.groups
        d['seq'] = RC3.get_seq()
        return d


class RSPP(Rbase):
    def __init__(self, layer):
        self.spp = layer
    
    def _render(self):
        conv = self.spp.cv1
        d = {}
        
        d['in_channels'] = conv.conv.in_channels
        d['out_channels'] = conv.conv.out_channels
        d['kernel_size'] = conv.conv.kernel_size
        d['stride'] = conv.conv.stride
        d['padding'] = conv.conv.padding
        d['bias'] = conv.conv.bias
        d['dilation'] = conv.conv.dilation
        d['groups'] = conv.conv.groups
        
        d['pool_kernel_size'] = [k.kernel_size for k in self.spp.m]
        d['seq'] = RSPP.get_seq()
        return d


class RConcat(Rbase):
    def __init__(self, layer):
        self.concat = layer
    
    def _render(self):
        d = {}
        d['dimension'] = self.concat.d 
        d['seq'] = RConcat.get_seq()
        return d


class RDetect(Rbase):
    def __init__(self, layer):
        self.detect = layer
        
    def _render(self):
        d = {}
        
        d['nc'] = self.detect.nc
        d['no'] = self.detect.no
        d['nl'] = self.detect.nl
        d['na'] = self.detect.na
        d['anchor'] = [v.item() for v in self.detect.anchors.view(-1)]
        d['anchor_len'] = len(self.detect.anchors.view(-1))
        d['inplace'] = self.detect.inplace
        d['seq'] = RDetect.get_seq()
        return d


class Model(object):
    
    def __init__(self, yolo_model):
        self.model = yolo_model
    
    def render(self, env, fobj):
        for layer in self.model:
            p_yolo_layer, name = is_yolo_layer(layer)
            if p_yolo_layer:
                instance = Registry.entries[name](layer)
                instance.render(env, fobj)
        template = env.get_template('Model.new.tpl')
        with open('model.test.cpp', 'w') as f:
            f.write(template.render())


