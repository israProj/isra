import os
import numpy as np
import onnx
import onnxruntime as rt
from settings import TEST_TVM
from settings import ITER_NUM

if TEST_TVM:
	from run_tvm import *

gen_model_saved = None
runner_output = None

graph_num = 0

error_num = 0
error_config = None
err_message_set = set()

def test(model_data, model_saved_file="tmp.onnx"):
	model, inputs_feed, network_node_num = model_data
	# print('The graph in model:\n{}'.format(model.graph))

	filename = model_saved_file
	onnx.save(model, filename)

	global graph_num
	global gen_model_saved
	graph_num += 1
	saved_file = gen_model_saved + "/g"+str(graph_num)+".onnx"
	print("saved:" + saved_file)
	onnx.save(model, saved_file)
	np.save(gen_model_saved + "/g%d_inputs.npy" % graph_num, inputs_feed)

	fix_dec = 1

	np.save("inputs.npy", inputs_feed)
	sess = rt.InferenceSession(filename)
	output_name = sess.get_outputs()[0].name
	onnxrt_out = sess.run([output_name], inputs_feed)
	print('onnx runtime finish normally!')
	
	out = np.around(onnxrt_out, decimals=fix_dec)

	# print('out=', out)

	tvm_out = {}
	if TEST_TVM:
		# i = ran(0, 4)
		for i in range(0, 4):
			tvm_out[i] = run_tvm(model, inputs_feed, i)
		
	# differential testing

	for i in tvm_out.keys():
		tvm_ver = 'tvm_with_opt_%d' % i
		out_deploy = np.around(tvm_out[i], decimals=fix_dec)

		if np.isnan(out).any() or np.isnan(out_deploy).any():
			return
		
		res = (out == out_deploy)
		res = np.array(res)

		if not (np.sum(res==True) >= res.size * 0.9):
			print(out)
			print(out_deploy)
			raise Exception("differential_testing: different results on onnxrt and %s!" % tvm_ver)


from difflib import SequenceMatcher
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_bugs():
	global error_num
	global err_message_set
	global runner_output

	crash_num = 0

	for iter in range(ITER_NUM):
		print("ITER=%d/%d" % (iter + 1, ITER_NUM))

		try:
			test(work())
			if DEBUG and (iter % 10 == 0):
				print("iter=", iter)
				print("OK!")
		except Exception as err:
			err_m = str(err)

			err_pd = err_m
			PROG_MES = 'Error(s) have occurred. The program has been annotated with them:'
			if PROG_MES in err_pd:
				err_pd = err_pd[:err_pd.find(PROG_MES)]
			if err_pd in err_message_set:
				print('............\n............\nsame error message!')
				continue
			dup = 0
			for err_m2 in err_message_set:
				if similarity(err_m2, err_pd) >= 0.8:
					dup = 1
					break
			if dup:
				print('............\n............\nprobably duplicated error message!')
				continue

			if not ('differential_testing' in err_pd):
				crash_num += 1
				err_message_set.add(err_pd)

			print(err_m)

			error_num += 1
			print("Find Bugs! Number %d" % error_num)
			print("............\n............\n............\n............\n")
			# print("error=", err_m)
			model = onnx.load("tmp.onnx")
			onnx.save(model, runner_output + "/bug%d.onnx" % error_num)
			inputs_feed = np.load("inputs.npy", allow_pickle=True)
			np.save(runner_output + "/bug%d_inputs.npy" % error_num, inputs_feed)
			with open(runner_output + "/bug%d_log.txt" % error_num, "w") as f:
				# print("tvm_params =", error_config, file=f)
				print("graph_num = ", graph_num)
				print("error_log = \n", err, file=f)

	print('iter=%d, all=%d, crash=%d' % (ITER_NUM, error_num, crash_num))


if __name__ == '__main__':
	from g import work
	gen_model_saved = 'genmodel-g-tmp'
	runner_output = "res_" + gen_model_saved

	OVERWRITE = True

	if os.path.exists(gen_model_saved):
		print('genmodel folder exists!')
		if not OVERWRITE:
			exit(0)
	else:
		os.mkdir(gen_model_saved)

	if os.path.exists(runner_output):
		print('output folder exists!')
		if not OVERWRITE:
			exit(0)
	else:
		os.mkdir(runner_output)

	find_bugs()
