

"""
SPP
Upsample
Concat
Detect
"""

class render_mixin(object):
    pass


class RFocus(object, render_mixin):
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
        
        
class RConv(object, render_mixin):
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
    
    
class RC3(object, render_mixin):
    def __init__(self, layer):
        self.c3 = layer
        
    def render(self):
        conv1 = self.c3.cv1
        
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
        shortcut = c3.m[0].add
        groups =  c3.m[0].cv2.conv.groups
        

class RSPP(object, render_mixin):
    def __init__(self, layer):
        self.spp = layer
    
    def render(self):
        return