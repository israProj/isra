import criterion
import onnx
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str,
                    help="pickrate/baseline", default='baseline')
parser.add_argument('--pickrate', type=float,
                    help="pickExistRate", default=0.95)
parser.add_argument('--type', type=str, help="dice/rand/inc", default='dice')
parser.add_argument('--minnode', type=int, default=1)
parser.add_argument('--maxnode', type=int, default=30)
parser.add_argument('--loop', type=int, default=1)
parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=1)
args = parser.parse_args()

with open("./bc.json", 'r') as load_f:
    block_corpus = json.load(load_f)
for loop in range(1, 6):
    f_name = './models/'+args.case+'/'+args.type+'/'+str(args.minnode)+'_'+str(args.maxnode) + \
        '_' + str(args.pickrate)+'/'+str(loop)+'/' +\
        args.type+'_'+str(args.minnode)+'_'+str(args.maxnode) + \
        '_'+str(args.pickrate)+'_e'+str(loop)+'.txt'
    f_timename = './models/'+args.case+'/'+args.type+'/'+str(args.minnode)+'_'+str(args.maxnode) + \
        '_' + str(args.pickrate) + '/time_'+args.type+'_'+str(args.minnode)+'_'+str(args.maxnode) + \
        '_' + str(args.pickrate)+'_e'+str(loop)+'.txt'
    if args.case == 'pickrate':
        f_name = './models/pickrate/'+str(+args.pickrate)+'/'+str(loop)+'/dice_'+str(args.minnode)+'_' +\
            str(args.maxnode)+'_'+str(args.pickrate)+'_e'+str(loop)+'.txt'
        f_timename = './models/pickrate/time/time_dice_'+str(args.minnode)+'_'+str(args.maxnode)+'_' +\
            str(args.pickrate)+'_e'+str(loop)+'.txt'

    if os.path.exists(f_name):
        os.remove(f_name)

    f_time = open(f_timename, 'r')
    test_set = {}
    g_metrics = []
    for iter in range(args.start, args.end+1):
        model = onnx.load(f_name[:-4]+'_'+str(iter)+'.onnx')
        ret_op, ret_I, g_metrics, test_set = criterion.get_coverage(
            [model], test_set, g_metrics, block_corpus)

        f = open(f_name, 'a')
        line = f_time.readline()
        f.write(line)
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
            'g', g_metrics[iter-1][0], g_metrics[iter-1][1], g_metrics[iter-1][2], g_metrics[iter-1][3], g_metrics[iter-1][4]))
        g_m = [sum(g[i] for g in g_metrics)/len(g_metrics)
               for i in range(5)]

        f.write('{:<10}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}\n'.format(
            'I', g_m[0], g_m[1], g_m[2], g_m[3], g_m[4]))
        f.write('\n')
        f.close()
    f_time.close()
