from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime

TEST_GPU = False

def run_tvm(model, inputs_feed, point_opt_level=None):
	onnx_model = model
	inputs = inputs_feed
	
	shape_dict = {}
	for k, v in inputs.items():
		shape_dict[k] = v.shape

	mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
	
	if point_opt_level is None:
		opt_level = ran(0, 4)
		# opt_level = 3
	else:
		opt_level = point_opt_level
	# print("opt_level=%d" % opt_level)
	
	target = "llvm"
	
	if TEST_GPU:
		target = "cuda"
	
	# error_config = [opt_level, target]
	
	with relay.build_config(opt_level=opt_level):
		tvm_graph, tvm_lib, tvm_params = relay.build_module.build(mod, target, params=params)
	
	ctx = tvm.cpu(0)
	if TEST_GPU:
		ctx = tvm.gpu()
	
	module = graph_runtime.create(tvm_graph, tvm_lib, ctx)
	module.load_params(relay.save_param_dict(tvm_params))
	
	for k, v in inputs.items():
		module.set_input(k, v)
	
	module.run()
	out_deploy = module.get_output(0).asnumpy()
	
	return out_deploy
