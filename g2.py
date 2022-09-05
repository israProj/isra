import os
import collections
import time
import numpy as np
import collections
from collections import deque
import json
import onnx
import g
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

import criterion
from settings import *
from gen_random import *


if __name__ == '__main__':
    test_set = {}
    g_metrics = []

    with open('./muffin_models.json', 'r') as size_f:
        sizes = json.load(size_f)
    f_name = './models/muffin/dice_'+str(pickExistRate)+'.txt'
    savesuffix = '.onnx'

    if os.path.exists(f_name):
        os.remove(f_name)

    with open("./bc.json", 'r') as load_f:
        full_block_corpus = json.load(load_f)
    with open('./commonops.json', 'r') as load_o:
        common = json.load(load_o).keys()
    block_corpus = {}
    for op in common:
        block_corpus[op] = full_block_corpus[op]

    errs = []
    total_time = 0.
    err_num = 0
    iter = 0
    for val in sizes.values():
        try:
            g.MIN_NODE = val
            g.MAX_NODE = val
            start_time = time.perf_counter()
            model, _, _ = g.work()
            end_time = time.perf_counter()
            total_time += (end_time-start_time)

            onnx.save(model, './models/muffin/dice/dice_' +
                      str(pickExistRate)+'_'+str(iter)+savesuffix)

            ret_op, ret_I, g_metrics, test_set = criterion.get_coverage(
                [model], test_set, g_metrics, block_corpus)

            f = open(f_name, 'a')
            f.write('at time ' + str(total_time) + ' generate ' +
                    str(iter+1) + ' valid models\n')
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

            # f.write(str(test_set)+'\n')
            f.write('\n')
            f.close()
            iter += 1
        except Exception as err:
            err_m = str(err)
            print("num-" + str(err_num) + " ERR=", err_m)
            err_num += 1
            continue
