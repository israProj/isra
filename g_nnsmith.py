import onnx
import json
import os
# import netron
import numpy as np
from onnx import shape_inference
# import criterion
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed, make_input_shape_fixed
import argparse
import shutil

import time
import criterion
import onnx
import json
import os
import argparse
import g
from settings import *
from gen_random import *

model_list = []
test_set = {}
g_metrics = []
iter = 0

with open('./nnsmith_models.json', 'r') as size_f:
    sizes = json.load(size_f)

f_name = './models/nnsmith/dice_'+str(pickExistRate)+'.txt'
if os.path.exists(f_name):
    os.remove(f_name)

with open("./bc.json", 'r') as load_f:
    full_block_corpus = json.load(load_f)
block_corpus = {}
with open('./nnsmithops.json', 'r') as load_o:
    commops = json.load(load_o).keys()
for op in commops:
    block_corpus[op] = full_block_corpus[op]

total_time = 0.0

for nnsmithsize in sizes.values():
        g.MIN_NODE = nnsmithsize
        g.MAX_NODE = nnsmithsize
        start_time = time.perf_counter()
        model, _, _ = g.work()
        end_time = time.perf_counter()
        total_time += (end_time-start_time)

        onnx.save(model, './models/nnsmith/dice_'+
                        str(pickExistRate)+'_'+str(iter)+'.onnx')

        ret_op, ret_I, g_metrics, test_set = criterion.get_coverage(
            [model], test_set, g_metrics, block_corpus)

        f = open(f_name, 'a')
        
        f.write(f'at time {total_time} generate {iter+1} valid models\n')
        f.write('{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format(
            'Object', 'OTC', 'IDC', 'ODC', 'SEC', 'DEC', 'SAC'))
        # for key in ret_op:
        #     if ret_op[key][0] == 0:
        #         f.write('{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format(
        #             key, '-', '-', '-', '-', '-', '-'))
        #     else:
        #         f.write('{:<20}{:<10.2%}{:<10.2%}{:<10.2%}{:<10.2%}{:<10.2%}{:<10.2%}\n'.format(
        #             key, ret_op[key][0], ret_op[key][1], ret_op[key][2], ret_op[key][3], ret_op[key][4], ret_op[key][5]))

        f.write('{:<20}{:<10.2%}{:<10.2%}{:<10.4f}{:<10.2%}{:<10.2%}{:<10.4f}\n'.format(
            'I', ret_I[0], ret_I[1], ret_I[2], ret_I[3], ret_I[4], ret_I[5]))
        op_names, op_nums = '', ''
        for key in ret_op:
            op_names += (key+' ')
            op_nums += (str(ret_op[key][6]) + ' ')
        f.write(op_names+'\n')
        f.write(op_nums+'\n')
        f.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format(
            'Object', 'NOO', 'NOT', 'NOP', 'NTR', 'NSA'))
        f.write('{:<10}{:<10}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}\n'.format(
            'g', g_metrics[iter][0], g_metrics[iter][1], g_metrics[iter][2], g_metrics[iter][3], g_metrics[iter][4]))
        g_m = [sum(g[i] for g in g_metrics)/len(g_metrics)
            for i in range(5)]

        f.write('{:<10}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}\n'.format(
            'I', g_m[0], g_m[1], g_m[2], g_m[3], g_m[4]))
        iter += 1

        f.write('\n')
        f.close()

print(iter)