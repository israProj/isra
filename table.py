import xlwt
from xlwt.Worksheet import Worksheet
import argparse
import numpy as np
from decimal import Decimal
import json 
import math

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int,  help="model_interval", default=250)
parser.add_argument('--total', type=int,  help="total_models", default=10000)
parser.add_argument('--type', type=str, help='pickrate/baseline/muffin', required=True)
parser.add_argument('--minnode', type=int,default=1)
parser.add_argument('--maxnode', type=int,default=50)
parser.add_argument('--pickrate', type=str,default='0.95')
args = parser.parse_args()

iter=1 if args.type=='muffin' else 5
pickrates = ['0.5', '0.8', '0.9', '0.95', '0.96', '0.97', '0.98', '0.99']
op_metric_names = ['OTC', 'IDC', 'ODC', 'SEC', 'DEC', 'SAC']
graph_metric_names = ['NOO', 'NOT', 'NOP', 'NTR', 'NSA']

model_interval = args.interval
total_model = args.total
commonops = []
rand_models = 0

cases = ['dice', 'rand', 'inc']
   
if args.type == 'muffin':
    cases = ['dice', 'muffin']
    op_cnts = [{} for c in range(len(cases))]
    with open('./commonops.json', 'r') as bc_f:
        bc = json.load(bc_f)
        for op_name in bc.keys():
            commonops.append(op_name)
            for c in range(len(cases)):
                op_cnts[c][op_name] = 0   
else:
    if args.type == 'pickrate':
        cases = ['0.5', '0.8', '0.9', '0.95', '0.96', '0.97', '0.98', '0.99']
    op_cnts = [{} for c in range(len(cases))]
    with open('./bc.json', 'r') as bc_f:
        bc = json.load(bc_f)
        for op_name in bc.keys():
            commonops.append(op_name)
            for c in range(len(cases)):
                op_cnts[c][op_name] = 0       
    

times = [[] for c in range(len(cases))]
op_ms = [[[] for c in range(len(cases))] for t in range(len(op_metric_names))]
g_ms = [[[] for c in range(len(cases))] for t in range(len(graph_metric_names))]
for j in range(len(cases)):
    print(cases[j])
    for i in range(1,iter+1):
        path = './models/'+args.type+'/'+cases[j]+'/'+str(args.minnode)+'_'+str(args.maxnode)+'_'+args.pickrate+\
            '/'+str(i)+'/'+cases[j]+'_'+str(args.minnode)+'_'+str(args.maxnode)+'_'+args.pickrate+'_e'+str(i)+'.txt' 
            
        if args.type == 'muffin':
            path = './models/muffin/'+cases[j]+'_'+args.pickrate+'.txt'
        elif args.type == 'pickrate':
            path = './models/pickrate/'+cases[j]+'/'+str(i)+'/dice_'+str(args.minnode)+'_'+str(args.maxnode)+\
            '_'+cases[j]+'_e'+str(i)+'.txt'

        f = open(path, 'r')
        lines = f.readlines()

        for k in range(1, 1+len(lines) // 9):
            if k %model_interval == 0 or k == total_model:      
                curno = math.ceil(k/model_interval)                  
                curlines = lines[(k-1)*9:(k)*9]
                # print(curlines[0])
                time = Decimal(curlines[0].split(' ')[2])
                # print(k, curno, len(times[j]))
                if len(times[j])<curno:
                    times[j].append(time)
                else:
                    times[j][curno-1]=times[j][curno-1]+time

                I_metric = [x for x in curlines[2].split(' ') if x != '']
                g_metric = [x for x in curlines[7].split(' ') if x != '']
                for o in range(len(op_metric_names)):
                    if len(op_ms[o][j])<curno:
                        op_ms[o][j].append(round(float(I_metric[1+o][:-1])/100, 4))
                    else:
                        op_ms[o][j][curno-1]+=round(float(I_metric[1+o][:-1])/100, 4)                       
                for o in range(len(graph_metric_names)):
                    if len(g_ms[o][j])<curno:
                        g_ms[o][j].append(float(g_metric[1+o]))
                    else:
                        g_ms[o][j][curno-1]+=float(g_metric[1+o])                     

        op_names = lines[-6].split(' ')[:-1]
        op_nums = lines[-5].split(' ')[:-1]
        for k, op in enumerate(op_names):
            op_cnts[j][op] += int(op_nums[k])
        if cases[j] == 'rand':
            rand_models += int(lines[-9].split(' ')[9])

    for k in range(len(times[j])):
        times[j][k] = times[j][k] / Decimal(str(iter))
    for o in range(len(op_metric_names)):
        for k in range(len(op_ms[o][j])):
            op_ms[o][j][k] /= iter
    for o in range(len(graph_metric_names)):
        for k in range(len(g_ms[o][j])):
            g_ms[o][j][k] /= iter
    for o in op_cnts[j].keys():
        op_cnts[j][o] /= iter
    # print(op_ms)

workbook = xlwt.Workbook(encoding = 'ascii')
model_xs = [x*model_interval for x in range(1, math.floor(total_model/model_interval)+1)]
  
if model_xs[-1]!=total_model:
    model_xs.append(total_model)
for i in range(len(op_metric_names)):
    worksheet = workbook.add_sheet(op_metric_names[i])
    worksheet.write(0,0, label = '# model')
    for t in range(len(cases)):
        worksheet.write(0,1+t, label = cases[t])
        for j, tm in enumerate(model_xs):
            worksheet.write(j+1,1+t, label = op_ms[i][t][j])
    for j, tm in enumerate(model_xs):
        worksheet.write(j+1,0, label = str(tm))    

for i in range(len(graph_metric_names)):
    worksheet = workbook.add_sheet(graph_metric_names[i])
    worksheet.write(0,0, label = '# model')
    for t in range(len(cases)):
        worksheet.write(0,1+t, label = cases[t])
        for j, tm in enumerate(model_xs):
            worksheet.write(j+1,1+t, label = g_ms[i][t][j])
    for j, tm in enumerate(model_xs):
        worksheet.write(j+1,0, label = str(tm))  

worksheet=workbook.add_sheet('time')
worksheet.write(0,0, label = '# model')
for t in range(len(cases)):
    worksheet.write(0,1+t, label = cases[t])
    for j, tm in enumerate(model_xs):
        worksheet.write(j+1,1+t, label = times[t][j])
for j, tm in enumerate(model_xs):
    worksheet.write(j+1,0, label = str(tm))  

worksheet = workbook.add_sheet('opcnt')
for i, op in enumerate(commonops):
    worksheet.write(1+i, 0, label = op)
for t in range(len(cases)):
    worksheet.write(0,1+t, label = cases[t])
    worksheet.write(1+len(commonops),1+t,label=sum(op_cnts[t].values()))
    worksheet.write(1+len(commonops)+1,1+t,label=total_model)
    for i, op in enumerate(commonops):
        worksheet.write(1+i, 1+t, label = op_cnts[t][op] / sum(op_cnts[t].values()))
worksheet.write(1+len(commonops),0, label='total_op')
worksheet.write(1+len(commonops)+1,0,label='valid model nums')
if args.type == 'baseline':
    worksheet.write(1+len(commonops)+2,0,label='total model nums')
    worksheet.write(1+len(commonops)+2,1,label=str(args.total))
    worksheet.write(1+len(commonops)+2,2,label=str(args.total))
    worksheet.write(1+len(commonops)+2,3,label=str(rand_models/iter))

if args.type == 'baseline':
    workbook.save('./results/metric_'+str(args.minnode)+'_'+str(args.maxnode)+'_'+args.pickrate+'.xls')
elif args.type == 'muffin':   
    workbook.save('./results/metric_'+args.pickrate+'_muffin.xls')
else:
    workbook.save('./results/metric_pickrate.xls')