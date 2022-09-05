import os
import collections
import time
import numpy as np
import collections
from collections import deque
import json
# import fpectl
from decimal import Decimal

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime as rt

import criterion
from settings import *
from gen_random import *

error_num = 0
error_config = None
err_message_set = set()

graph_num = 0
gen_model_saved = "gen_model-g_inc-tmp"

def work():
	tensor_list = []
	tensor_map = {}
	tensor_init = {}
	tensor_init_type = {}
	node_list = []
	input_tensor = []
	inputs_feed = {}
	output_tensor = []
	init_tensor = []
	pure_value_tensor = []

	if not REPLAY:
		global rand_record
		rand_record = []

	network_node_num = ran(MIN_NODE, MAX_NODE + 1)

	ops_seq = None

	global global_tensor_num
	global_tensor_num = 0

	def rand_shape():
		fixed_dim = ran(MIN_TENSOR_DIM, MAX_TENSOR_DIM + 1)
		fixed_shape = [ran(1, MAX_TENSOR_DIM_LEN + 1) for i in range(fixed_dim)]
		return fixed_shape

	def new_tensor(shape, data_type=TensorProto.FLOAT, data_value=None):
		global global_tensor_num
		global_tensor_num += 1
		cur_name = 'node' + str(global_tensor_num)
		cur_tensor = helper.make_tensor_value_info(cur_name, data_type, shape)
		if (data_value is None) and (-1 not in shape):
			if data_type == TensorProto.FLOAT:
				cur_value = ran_input(shape)
				cur_value = cur_value * 2 - 1
				cur_value = cur_value.astype(np.float32)
		else:
			cur_value = data_value
		tensor_list.append(cur_name)
		tensor_map[cur_name] = cur_tensor
		tensor_init[cur_name] = cur_value
		tensor_init_type[cur_name] = data_type
		return cur_name

	def tensor_shape(t_name):
		return list(tensor_init[t_name].shape)

	def pass_value(t, given_value=True):
		pure_value_tensor.append(t)
		t_value = tensor_init[t]
		t_type = tensor_init_type[t]
		t = tensor_map[t]

		if NO_INPUT or given_value:
			init_tensor.append(helper.make_tensor(t.name, t_type, dims=t_value.shape, vals=t_value.flatten()))
		else:
			input_tensor.append(t)
			inputs_feed[t.name] = t_value

	first_n = new_tensor(rand_shape())
	pass_value(first_n, False)

	dq = []
	dq.append(first_n)

	no_succ = set()
	no_succ.add(first_n)

	for step in range(network_node_num):
		n_iter = 0
		success = True
		while True:
			n_iter += 1
			if n_iter > 10000:
				assert("try too many times!")
				success = False
				break

			v = ran(0, len(ops))
			new_node_type = ops[v]
			if ops_seq != None:
				new_node_type = ops_seq[step % len(ops_seq)]
			node_name = 'op' + str(step)
			kwargs = {}

			# print("gen op=%s..\n\n\n" % new_node_type)

			# ---------------------------
			def getNewT():
				if ran(0, 100000) < 100000 * pickExistRate:
					return dq[ran(0, len(dq))]
				else:
					newT = new_tensor(rand_shape())
					pass_value(newT)
					dq.append(newT)
					return newT

			n1 = getNewT()
			n1_shape = tensor_shape(n1)
			n1_dim = len(n1_shape)

			out_shape = tensor_shape(n1)
			outputs = []

			if new_node_type in ['Softmax', 'LpNormalization', 'Concat', 'Compress', 'Flatten']:
				kwargs['axis'] = ran(0, len(tensor_shape(n1)))
			if new_node_type in reduce_ops:
				if new_node_type != 'ReduceSum':
					kwargs['axes'] = ran_ord(n1_dim)[:ran(1, n1_dim + 1)]

			if new_node_type == 'SpaceToDepth':
				t = ran(1, MAX_TENSOR_DIM_LEN + 1)
				kwargs['blocksize'] = t

			if new_node_type in ['MaxPool', 'AveragePool', 'LpPool']:
				kwargs['kernel_shape'] = [ran(1, MAX_TENSOR_DIM_LEN + 1) for i in range(n1_dim - 2)]
				kwargs['strides'] = [ran(1, MAX_TENSOR_DIM_LEN + 1) for i in range(n1_dim - 2)]

			if new_node_type in ['Conv', 'ConvTranspose']:
				kwargs['kernel_shape'] = [ran(1, MAX_TENSOR_DIM_LEN + 1) for i in range(n1_dim - 2)]	
				kwargs['strides'] = [ran(1, MAX_TENSOR_DIM_LEN + 1) for i in range(n1_dim - 2)]
				kwargs['pads'] = [0 for i in range((n1_dim - 2) * 2)]
				if ran(0, 2) == 0:
					kwargs['group'] = 1

			if new_node_type in ['LpPool', 'LpNormalization']:
				if ran(0, 2) == 1:
					kwargs['p'] = ran(1, 3)
			if new_node_type in ['LeakyRelu']:
				if ran(0, 2) == 1:
					kwargs['alpha'] = ran(1, 3) * 0.01
			if new_node_type in ['Elu', 'ThresholdedRelu']:
				if ran(0, 2) == 1:
					kwargs['alpha'] = 1.0 * ran(1, 3) / ran(1, 3)
			if new_node_type == 'Transpose':
				x_ord = ran_ord(n1_dim)
				if ran(0, 100) > 0:
					kwargs['perm'] = x_ord

			if new_node_type == 'Split':
				ax = 0
				if ran(0, 2) > 0:
					ax = kwargs['axis'] = ran(0, n1_dim)

				n2_dim = ran(1, MAX_TENSOR_DIM + 1)
				n2_shape = [n2_dim]
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array([ran(1, MAX_TENSOR_DIM_LEN) for i in range(n2_dim)]))
				pass_value(n2)

				inputs = [n1, n2]
				nums_output = ran(1, MAX_MULTI_OUTPUTS + 1)
				out_n = []
				for i in range(nums_output):
					out_n_shape = [ran(1, MAX_TENSOR_DIM_LEN + 1) for i in range(n1_dim)]
					out_n.append(new_tensor(out_n_shape))
					outputs.append(out_n[i])

			elif new_node_type in ['GlobalMaxPool', 'GlobalAveragePool']:
				inputs = [n1]
			elif new_node_type == 'Size':
				inputs = [n1]
			elif new_node_type == 'ReduceSum':
				n2_dim = 1
				n2_value = ran_ord(n1_dim)[:ran(1, n1_dim + 1)]
				n2_shape = [len(n2_value)]
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(n2_value))
				pass_value(n2)
				inputs = [n1, n2]
			elif new_node_type == 'Tile':
				n2_dim = 1
				n2_shape = [n1_dim]
				tile_value = [ran(1, 4) for i in range(n1_dim)]
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(tile_value))
				pass_value(n2)
				inputs = [n1, n2]
			elif new_node_type == 'Gather':
				ax = 0
				if ran(0, 10) > 0:
					kwargs['axis'] = ran(0, n1_dim)
					ax = kwargs['axis']
				indices_dim = ran(1, MAX_TENSOR_DIM - n1_dim + 2)
				indices_shape = []
				tot_c = 1
				for i in range(indices_dim):
					if ran(0, 2) > 0:
						indices_shape.append(ran(1, MAX_TENSOR_DIM_LEN + 1))
					else:
						indices_shape.append(1)
					tot_c = tot_c * indices_shape[i]
				all_indices = [ran(0, n1_shape[ax]) for i in range(tot_c)]
				indices_value = np.array(all_indices).reshape(indices_shape)

				indices = new_tensor(indices_shape, TensorProto.INT64, np.array(indices_value))
				pass_value(indices)

				inputs = [n1, indices]
			elif new_node_type == 'Slice':
				starts_shape = [n1_dim]
				starts_value = [ran(0, n1_shape[i]) for i in range(n1_dim)]
				ends_shape = [n1_dim]
				ends_value = [starts_value[i] + ran(0, n1_shape[i] - starts_value[i]) + 1 for i in range(n1_dim)]
				axes_shape = [n1_dim]
				axes_value = [i for i in range(n1_dim)]
				steps_shape = [n1_dim]
				steps_value = [ran(1, 4) for i in range(n1_dim)]

				starts = new_tensor(starts_shape, TensorProto.INT64, np.array(starts_value))
				pass_value(starts)
				ends = new_tensor(ends_shape, TensorProto.INT64, np.array(ends_value))
				pass_value(ends)
				axes = new_tensor(axes_shape, TensorProto.INT64, np.array(axes_value))
				pass_value(axes)
				steps = new_tensor(steps_shape, TensorProto.INT64, np.array(steps_value))
				pass_value(steps)
							
				inputs = [n1, starts, ends, axes, steps]
			elif new_node_type == 'Resize':
				scales_dim = 1
				scales_shape = [n1_dim]
				scales = new_tensor(scales_shape, TensorProto.FLOAT, np.array([1.0 for i in range(n1_dim)]))
				pass_value(scales)

				roi_dim = 1
				roi_shape = [2 * n1_dim]
				roi_value = []
				for i in range(n1_dim):
					roi_value.append(0)
				for i in range(n1_dim):
					roi_value.append(1)
				roi = new_tensor(roi_shape, TensorProto.FLOAT, np.array([roi_value]))
				pass_value(roi)

				kwargs['mode'] = 'nearest'
				inputs = [n1, roi, scales]
			elif new_node_type == 'Reshape':
				def gcd(a, b):
					return (a if b == 0 else gcd(b, a % b)) 
				t = 1
				for i in range(n1_dim):
					t = t * n1_shape[i]
				new_dim = ran(1, MAX_TENSOR_DIM + 1)
				reshape_shape = []
				for i in range(new_dim - 1):
					t2 = gcd(t, ran(1, t + 1))
					t = t // t2
					reshape_shape.append(t2)
				reshape_shape.append(t)

				n2_dim = 1
				n2_shape = [new_dim]
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(reshape_shape))
				pass_value(n2)

				inputs = [n1, n2]
			elif new_node_type == 'Unsqueeze':
				expanded_dim = ran(1, MAX_TENSOR_DIM + 1)
				x_ord = ran_ord(expanded_dim)
				n2_dim = 1
				n2_shape = [expanded_dim - n1_dim]
				n2_value = []
				for i in range(expanded_dim - n1_dim):
					n2_value.append(x_ord[i])
				n2_value = sorted(n2_value)
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(n2_value))
				pass_value(n2)

				inputs = [n1, n2]
			elif new_node_type == 'Expand':
				n2_dim = 1
				new_dim = n1_dim + ran(0, MAX_TENSOR_DIM - n1_dim + 1)
				new_dim = n1_dim 
				n2_shape = [new_dim]
				expand_shape = [n1_shape[i] for i in range(n1_dim)]
				for i in range(new_dim - n1_dim):
					expand_shape = [1] + expand_shape
				saved_shape = [expand_shape[i] for i in range(new_dim)]
				for i in range(new_dim):
					if expand_shape[i] == 1:
						if ran(0, 2) == 0:
							expand_shape[i] = ran(2, 5)
					else:
						expand_shape[i] = 1

				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(expand_shape))
				pass_value(n2)
				inputs = [n1, n2]
			elif new_node_type == 'Pad':
				pads_value = []
				kwargs['mode'] = 'constant'
				for i in range(n1_dim):
					pad_n = ran(0, 3)
					pads_value += [pad_n]
				for i in range(n1_dim):
					pad_n = ran(0, 3)
					pads_value += [pad_n]
				pads = new_tensor([2 * n1_dim], TensorProto.INT64, np.array(pads_value))
				pass_value(pads)

				inputs = [n1, pads]
			elif new_node_type == 'BatchNormalization':
				s = new_tensor(rand_shape())
				pass_value(s)
				bias = new_tensor(rand_shape())
				pass_value(bias)
				mean = new_tensor(rand_shape())
				pass_value(mean)
				var = new_tensor(rand_shape())
				pass_value(var)
				inputs = [n1, s, bias, mean, var]
			elif new_node_type == 'Compress':
				l = n1_shape[kwargs['axis']]
				n2_dim = 1
				n2_shape = [l]

				compress_shape = [ran(0, 2) for i in range(l)]
				compress_sum = int(sum(compress_shape))
				if compress_sum == 0:
					compress_shape[ran(0, l)] = 1
					compress_sum = 1
					
				n2 = new_tensor(n2_shape, TensorProto.BOOL, np.array(compress_shape))
				pass_value(n2)
				inputs = [n1, n2]
			elif new_node_type in extra_t_ops:
				n2 = getNewT()
				inputs = [n1, n2]
			else:
				inputs = [n1]

			if new_node_type in multi_extra_t_ops:
				extra_num = ran(0, MAX_MULTI_INPUTS - 1)
				for t in range(extra_num):
					n_another = getNewT()
					inputs.append(n_another)
			if new_node_type == "Concat":
				extra_num = ran(0, MAX_MULTI_INPUTS - 1)
				for t in range(extra_num):
					n_another = getNewT()
					inputs.append(n_another)

			if new_node_type not in ['Split']:
				out_tensor = new_tensor(out_shape)
				outputs = [out_tensor]

			new_node = helper.make_node(new_node_type, inputs=inputs, outputs=outputs, name=node_name, **kwargs)

			tmp_output_tensor = []
			for x in no_succ:
				if x in pure_value_tensor:
					continue
				if x in inputs:
					continue
				tmp_output_tensor.append(tensor_map[x])
			for x in outputs:
				tmp_output_tensor.append(tensor_map[x])

			node_list.append(new_node)
			tmp_graph_def = helper.make_graph(node_list, "tmp-test-model", input_tensor, tmp_output_tensor, init_tensor)

			tmp_model = helper.make_model(tmp_graph_def, producer_name='onnx-example')
			if not valid((tmp_model, None, None)):
				node_list.pop()
				continue

			for t in inputs:
				if t in no_succ:
					no_succ.remove(t)

			for x in outputs:
				no_succ.add(x)

			for x in outputs:
				dq.append(x)

			break




	def totElement(x):
		ans = 1
		for i in tensor_shape(x):
			ans = ans * i
		return ans

	output_ts = []
	tot_output_element = 0
	for x in no_succ:
		if x in pure_value_tensor:
			continue
		x2 = new_tensor([1, totElement(x)])
		output_ts.append(x2)
		n = helper.make_node('Flatten', inputs=[x], outputs=[x2], name='flatten_%s' % x, axis=0)
		node_list.append(n)
		tot_output_element += totElement(x)


	final_tensor = new_tensor([1, tot_output_element])	
	n = helper.make_node('Concat', inputs=output_ts, outputs=[final_tensor], name='concat_outputs', axis=-1)
	node_list.append(n)
	output_tensor.append(tensor_map[final_tensor])
	

	# output_tensor = []
	# for x in no_succ:
	# 	if x in pure_value_tensor:
	# 		continue
	# 	output_tensor.append(tensor_map[x])
	graph_def = helper.make_graph(node_list, "test-model", input_tensor, output_tensor, init_tensor)
	model = helper.make_model(graph_def, producer_name='onnx-example')

	return model, inputs_feed, network_node_num


def valid(model_data):
	model, inputs_feed, network_node_num = model_data
	try:
		sess = rt.InferenceSession(model.SerializeToString())
	except Exception as err:
		err_m = str(err)
		return False
	return True



if __name__ == '__main__':	

	# find_bugs()
	# # work()

	# with open('output/tmp.onnx_rec.txt', 'w') as f:
	# 	f.write(json.dumps(rand_record))
	# debug()


	errs = []
	total_time = 0.
	err_num = 0
	loop = args.loop
	total_time = 0.

	f_name = './models/baseline/inc/'+str(MIN_NODE)+'_'+str(MAX_NODE)+'_'+str(pickExistRate) +\
			'/time_inc_'+str(MIN_NODE)+'_'+str(MAX_NODE)+'_'+str(pickExistRate)+'_e'+str(loop)+'.txt'
	model_prefix = './models/baseline/inc/'+str(MIN_NODE)+'_'+str(MAX_NODE)+'_'+str(pickExistRate)+\
					'/'+str(loop)+'/inc_'+str(MIN_NODE)+'_'+str(MAX_NODE)+'_'+str(pickExistRate)+'_e'+str(loop)+'_'

	if os.path.exists(f_name):
		os.remove(f_name)

	def create_folder(s):
		folder_s = s[:len(s) - s[::-1].index('/')]
		if not os.path.isdir(folder_s):
			os.makedirs(folder_s)
	create_folder(model_prefix)
	create_folder(f_name)

	for iter in range(ITER_NUM):	
		try:
			start_time = time.perf_counter()
			model, _, _ = work()
			end_time = time.perf_counter()
			total_time += (end_time-start_time)
			onnx.save(model, model_prefix+str(iter+1)+'.onnx')

			f = open(f_name , 'a')
			f.write('at time '+ str(total_time) + ' generate '+ str(iter+1) +' valid models\n')
			f.close()

		except Exception as err:
			err_m = str(err)
			print("num-" + str(err_num) + " ERR=", err_m)
			err_num+=1
			# continue






	