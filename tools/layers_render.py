
import os


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
            Registry.entries[name] = cls


class Rbase(metaclass=Registry):
    count = 0
    
    @classmethod
    def get_seq(cls):
        cls.count += 1
        return cls.count
    
    def get_tpl_name(self):
        tpl_file_name = "{}.new.tpl".format(self.__class__.__name__[1:])
    

class RFocus(Rbase):
    
    def __init__(self, layer):
        self.focus = layer
        
    def render(self):
        conv = self.focus.conv
        
        in_channels = conv.conv.in_channels
        out_channels = conv.conv.out_channels
        kernel_size = conv.conv.kernel_size
        stride = conv.conv.stride
        padding = conv.conv.padding
        bias = conv.conv.bias
        dilation = conv.conv.dilation
        group = conv.conv.group
        

class RConv(Rbase):
    def __init__(self, layer):
        self.conv = layer
        
    def render(self):
        conv = self.conv 
        
        in_channels = conv.conv.in_channels
        out_channels = conv.conv.out_channels
        kernel_size = conv.conv.kernel_size
        stride = conv.conv.stride
        padding = conv.conv.padding
        bias = conv.conv.bias
        dilation = conv.conv.dilation
        group = conv.conv.group


class RC3(Rbase):
    def __init__(self, layer):
        self.c3 = layer
        
    def render(self):
        conv = self.c3.cv1
        
        in_channels = conv.conv.in_channels
        out_channels = conv.conv.out_channels
        kernel_size = conv.conv.kernel_size
        stride = conv.conv.stride
        padding = conv.conv.padding
        bias = conv.conv.bias
        dilation = conv.conv.dilation
        group = conv.conv.group
    
        n = len(self.c3.m)
        eps = out_channels/in_channels
        shortcut = self.c3.m[0].add
        groups =  self.c3.m[0].cv2.conv.groups


class RSPP(Rbase):
    def __init__(self, layer):
        self.spp = layer
    
    def render(self):
        
        conv = self.spp.cv1
        
        in_channels = conv.conv.in_channels
        out_channels = conv.conv.out_channels
        kernel_size = conv.conv.kernel_size
        stride = conv.conv.stride
        padding = conv.conv.padding
        bias = conv.conv.bias
        dilation = conv.conv.dilation
        group = conv.conv.group
        
        k = [k.kernel_size for k in self.spp.m]


class RConcat(Rbase):
    def __init__(self, layer):
        self.concat = layer
    
    def render(self):
        dimension = self.concat.d 


class RDetect(Rbase):
    def __init__(self, layer):
        self.detect = layer
        
    def render(self):
        nc = self.detect.nc
        no = self.detect.no
        nl_= self.detect.nl
        na = self.detect.na
        
        anchors = [v.item() for v in self.detect.anchors.view(-1)]


class Model(object):
    
    def __init__(self, yolo_model):
        self.model = yolo_model
    
    def _prepare_env(self):
        if os.pathexists("templates/workspace"):
            os.remove("templates/workspace")
        os.mkdir("templates/workspace")

    def render(self):
        self._prepare_env()
        for layer in self.model:
            p_yolo_layer, name = is_yolo_layer(layer)
            if p_yolo_layer:
                instance = Registry.entries[name](layer)
                instance.render()


