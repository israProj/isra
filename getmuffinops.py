import onnx
import json
import netron
import numpy as np
from onnx import shape_inference
import criterion
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed, make_input_shape_fixed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, help="start index", default=1)
parser.add_argument('--end', type=int,  help="end index", default=1)
parser.add_argument('--dir', type=str, help="dir to muffin models")
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
    input_feeds = np.load(args.dir +'/h'+str(i)+'.onnx.npy', allow_pickle=True).item()
    shape_dict = {}
    for k, v in input_feeds.items():
        make_input_shape_fixed(model.graph, k, np.array(v).shape)
    fix_output_shapes(model)
    infer = shape_inference.infer_shapes(model)
    onnx.save(infer, args.dir+'/h'+str(i)+'fix.onnx')   


newbc = json.load(open('./bc.json', 'r'))
muffin_ops = []
same_models = {}

ok = 0

for i in range(args.start, args.end+1):
    try:
        cnt = 0
        hasconcat = 0
        out = False

        model = onnx.load(args.dir+'/h'+str(i)+'fix.onnx')

        # filter ops not implemented
        for j, node in enumerate(model.graph.node):
            if node.op_type not in newbc.keys():
                out = True
                break
            if node.op_type == 'Concat':
                hasconcat +=1
        if out:
            continue

        # fix shape
        # fix_shape(i, model)

        # test shape
        # test_shape(model)

        # calculate size
        for model_out_i, output in enumerate(model.graph.output):
            for node in model.graph.node:
                for node_out_i, nodeout in enumerate(node.output):
                    if nodeout == output.name:
                        output.name = 'model_output_'+str(model_out_i)
                        node.output[node_out_i] = 'model_output_' + \
                            str(model_out_i)

        # for j, node in enumerate(model.graph.node):
        #     if not criterion.last_is_reshape(node):
        #         cnt += 1

        if cnt == 0 or out:
            continue

        ok += 1
        same_models[i] = cnt
        for j, node in enumerate(model.graph.node):
            if node.op_type not in muffin_ops:
                muffin_ops.append(node.op_type)

        if ok == 10000:
            break
    except Exception as err:
        # print(str(err))
        continue

with open('./muffin_models.json', 'w') as wf:
    json.dump(same_models, wf)

common ={}
with open('./muffinops.json', 'w') as wfo:
    for op in muffin_ops:
        if op in newbc.keys():
            common[op] = {}
    json.dump(common, wfo)

