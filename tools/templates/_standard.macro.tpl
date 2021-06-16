
{% macro conv2d(in_channels, out_channels, kernel_size, padding, bias, groups, stride) %} nn.Conv2d( {{ in_channels }}, {{ out_channels}}, {{ kernel_size| expand_arr_int}} ).bias({{bias|bool}}){{ padding | padding }}{{ stride | stride}}{{ groups | groups }} {% endmacro %}

