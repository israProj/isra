import onnx
import json
import netron
import numpy as np
from onnx import shape_inference
import criterion
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed, make_input_shape_fixed
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, help="start index", default=1)
parser.add_argument('--end', type=int,  help="end index", default=1)
parser.add_argument('--dir', type=str, help="dir to nnsmith models")
args = parser.parse_args()


def test_shape(i, model):
    for v in model.graph.value_info:
        for j in v.type.tensor_type.shape.dim:
            if j.dim_value == 0:
                print(i)
                return True
    for j in model.graph.initializer:
        for d in j.dims:
            if d == 0:
                print(i)
                return True
    for j in model.graph.input:
        for d in j.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                print(i)
                return True
    return False

def fix_shape(i, model):
    input_feeds = {}
    input_feeds = pickle.load(open(args.dir+'/'+str(i)+'.pkl','rb'))[0]
    shape_dict = {}
    for k, v in input_feeds.items():
        make_input_shape_fixed(model.graph, k, np.array(v).shape)
    fix_output_shapes(model)
    infer = shape_inference.infer_shapes(model)
    onnx.save(infer, args.dir+'/h'+str(i)+'fix.onnx')   


newbc = json.load(open('./bc.json', 'r'))
nnsmith_ops = set()
same_models = {}

ok = 0

for i in range(args.start, args.end+1):
    try:
        model = onnx.load(os.path.join(args.dir,str(i)+'.onnx'))
        nnsmithsize = 0 
        for j in model.graph.node:
            if j.op_type in newbc.keys():
                nnsmithsize += 1
            nnsmith_ops.add(j.op_type)
            

        # fix shape
        # fix_shape(i, model)

        # test shape
        # test_shape(model)
                
        ok += 1
        same_models[i] = nnsmithsize

        if ok == 10000:
            break
    except Exception as err:
        print(str(err))
        continue
print(same_models)
with open('./nnsmith_models.json', 'w') as wf:
    json.dump(same_models, wf)

common ={}
with open('./nnsmithops.json', 'w') as wfo:
    for op in nnsmith_ops:
        if op in newbc.keys():
            common[op] = {}
    json.dump(common, wfo)
print(len(common))
