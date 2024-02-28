import criterion
import onnx
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pickrate', type=float,
                    help="pickExistRate", default=0.95)
parser.add_argument('--dir', type=str, help="dir to muffin models")
args = parser.parse_args()

model_list = []
test_set = {}
g_metrics = []
iter = 0
rate = args.pickrate

f_name = './models/muffin/muffin_'+str(args.pickrate)+'.txt'
if os.path.exists(f_name):
    os.remove(f_name)

with open("./muffinops.json", 'r') as load_o:
    common = json.load(load_o).keys()
with open("./bc.json", 'r') as load_f:
    full_block_corpus = json.load(load_f)
block_corpus = {}
for op in common:
    block_corpus[op] = full_block_corpus[op]


modelpath = './muffin_models.json'
with open(modelpath, 'r') as size_f:
    size_list = json.load(size_f)

for ii in size_list.keys():
    print(ii)
    model = onnx.load(args.dir+'/h'+str(ii)+'fix.onnx')

    ret_op, ret_I, g_metrics, test_set = criterion.get_coverage(
        [model], test_set, g_metrics, block_corpus)

    f = open(f_name, 'a')
    f.write('at time 0 generate ' + str(iter+1) + ' valid models\n')
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
