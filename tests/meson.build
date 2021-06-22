project('libtorch-mnist', 'cpp', default_options : ['default_library=static', 'cpp_std=c++14'])

torch_inc = include_directories('/Users/xurui/Public/local/libtorch/include', '/Users/xurui/Public/local/libtorch/include/torch/csrc/api/include')

cpp = meson.get_compiler('cpp')
torch_dep = cpp.find_library('torch', dirs: '/Users/xurui/Public/local/libtorch/lib')
c10_dep = cpp.find_library('c10', dirs: '/Users/xurui/Public/local/libtorch/lib')
protobuf_dep = cpp.find_library('protobuf', dirs: '/Users/xurui/Public/local/libtorch/lib')
torch_cpu_dep = cpp.find_library('torch_cpu', dirs: '/Users/xurui/Public/local/libtorch/lib')


my_inc = include_directories('/Users/xurui/Git/yolov5-cpp/layers')
executable('app', 'model.test.cpp', include_directories: [torch_inc, my_inc], dependencies: [torch_dep, torch_cpu_dep, c10_dep, protobuf_dep])