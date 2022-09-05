# -*- coding: utf-8 -*-
import onnx
from onnx import shape_inference
import netron
import argparse
import json

# for operators, see https://github.com/onnx/onnx/blob/master/docs/Operators.md
# for proto, see https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def last_is_reshape(node):
    for output in node.output:
        if output.startswith('model_output_'):
            return 'Reshape' in node.name
    return False


def get_tensor_dim(edge, graph):
    dims, values = [], []
    in_values = False

    if edge.startswith('model_output_'):
        values = graph.output[int(edge[13:])]
    else:
        values = graph.value_info

    for v in values:
        if v.name == edge:
            for i in v.type.tensor_type.shape.dim:
                dims.append(i.dim_value)
                in_values = True
            break
    if not in_values:
        for i in graph.initializer:
            if i.name == edge:
                for d in i.dims:
                    dims.append(d)
                break
        for i in graph.input:
            if i.name == edge:
                for d in i.type.tensor_type.shape.dim:
                    dims.append(d.dim_value)
                break

    return dims


def get_metric_for_op(op, test_set, block_corpus):
    MAX_SPC = 1000

    metrics = [1]  # otc
    metrics.append(len(test_set[op]['input']) /
                   len(block_corpus[op]['in_degree']))  # idc
    metrics.append(len(test_set[op]['output']))  # odc
    metrics.append(len(test_set[op]['edgetype']) / len(block_corpus))  # sec
    metrics.append(len(test_set[op]['tripletype']) /
                   (len(block_corpus)*len(block_corpus)))  # dec
    metrics.append((len(test_set[op]['attrs']) +
                   len(test_set[op]['dims'])))  # spc
    metrics.append(test_set[op]['cnt'])
    return metrics


def get_edges_with_dims(model):
    inputs = []
    for input in model.graph.input:
        inputs.append(input.name)
    for init in model.graph.initializer:
        inputs.append(init.name)
    edges = {}
    for i, output in enumerate(model.graph.output):
        edges[output.name] = []
    for name in inputs:
        edges[name] = []
    dims = {}

    for i in range(len(model.graph.node)):
        end_node = model.graph.node[i]
        for edge in end_node.input:
            if edge not in dims and not last_is_reshape(end_node) \
                    and 'concat_outputs' not in end_node.name and 'flatten_node' not in end_node.name:
                dims[edge] = get_tensor_dim(edge, model.graph)
            if edge in inputs:
                edges[edge].append(
                    (-inputs.index(edge)-1-len(model.graph.output), i))
                continue

            for j in range(len(model.graph.node)):
                start_node = model.graph.node[j]
                if edge in start_node.output:
                    if edge in edges:
                        edges[edge].append((j, i))
                    else:
                        edges[edge] = [(j, i)]

        for edge in end_node.output:
            if edge.startswith('model_output_'):
                e_i = int(edge[13:])
                edges[edge].append((i, -1-e_i))

    return edges, dims, inputs


def parse_graph(model, test_set, block_corpus):
    INFINITE_INPUT = 5
    MAX_OUTPUT = 12
    newattrcnt = 0

    inferred_model = shape_inference.infer_shapes(model)
    edges, dims, inputs = get_edges_with_dims(inferred_model)

    for i, node in enumerate(model.graph.node):
        if 'flatten_node' in node.name or 'concat_outputs' in node.name:
            continue
        if last_is_reshape(node):
            continue
        if node.op_type in block_corpus:
            if node.op_type not in test_set:
                test_set[node.op_type] = {'input': [], 'output': [], 'edgetype': [
                ], 'tripletype': [], 'attrs': [], 'dims': [], 'cnt': 0}
            test_set[node.op_type]['cnt'] += 1
            for output in node.output:
                if output.startswith('model_output_'):
                    continue
                for edge in edges[output]:
                    node1 = inferred_model.graph.node[edge[1]]
                    if (node1.op_type in block_corpus) and not last_is_reshape(node1) \
                            and ('flatten_node' not in node1.name) and ('concat_outputs' not in node1.name):
                        if node1.op_type not in test_set[node.op_type]['edgetype']:
                            test_set[node.op_type]['edgetype'].append(
                                node1.op_type)
                        for output1 in node1.output:
                            if output1.startswith('model_output_'):
                                continue
                            for edge1 in edges[output1]:
                                node2 = inferred_model.graph.node[edge1[1]]
                                if (node2.op_type in block_corpus) and not last_is_reshape(node2) \
                                        and ('flatten_node' not in node2.name) and ('concat_outputs' not in node2.name):
                                    if (node1.op_type, node2.op_type) not in test_set[node.op_type]['tripletype']:
                                        test_set[node.op_type]['tripletype'].append(
                                            (node1.op_type, node2.op_type))

            if node.attribute and node.attribute not in test_set[node.op_type]['attrs']:
                test_set[node.op_type]['attrs'].append(node.attribute)
                newattrcnt += 1
            for input in node.input:
                if dims[input] not in test_set[node.op_type]['dims']:
                    test_set[node.op_type]['dims'].append(dims[input])

        suffix = []
        for j in node.output:
            if j.startswith('model_output_'):
                continue
            for k in range(len(edges[j]) - 1):
                suffix.append(j)
        node.output.extend(suffix)

    for i, node in enumerate(model.graph.node):
        if 'flatten_node' in node.name or 'concat_outputs' in node.name:
            continue
        if last_is_reshape(node):
            continue
        if node.op_type in block_corpus:
            inputlen = INFINITE_INPUT if len(
                node.input) > INFINITE_INPUT else len(node.input)
            if inputlen not in test_set[node.op_type]['input']:
                test_set[node.op_type]['input'].append(inputlen)
            outputlen = MAX_OUTPUT if len(
                node.output) > MAX_OUTPUT else len(node.output)
            if outputlen not in test_set[node.op_type]['output']:
                test_set[node.op_type]['output'].append(outputlen)

    return edges, test_set, dims, newattrcnt


def get_graph_metric(model, edges):
    cnt_op = 0
    pairs = []
    ops = []

    oppairs = []
    optriples = []
    for i in range(len(model.graph.node)):
        fromnode = model.graph.node[i]
        if 'flatten_node' in fromnode.name or 'concat_outputs' in fromnode.name:
            continue
        if last_is_reshape(fromnode):
            continue
        cnt_op += 1
        if fromnode.op_type not in ops:
            ops.append(fromnode.op_type)
        for tensors in edges.values():
            for edge in tensors:
                if edge[0] == i and edge[1] >= 0:
                    tonode = model.graph.node[edge[1]]
                    if 'flatten_node' not in tonode.name and 'concat_outputs' not in tonode.name \
                            and not last_is_reshape(tonode):
                        pair = (fromnode.name, tonode.name)
                        if pair not in pairs:
                            pairs.append(pair)
                        oppair = (fromnode.op_type, tonode.op_type)
                        if oppair not in oppairs:
                            oppairs.append(oppair)

                        for tensors2 in edges.values():
                            for edge2 in tensors2:
                                if edge2[0] == edge[1] and edge2[1] >= 0:
                                    tonode2 = model.graph.node[edge2[1]]
                                    if 'flatten_node' not in tonode2.name and 'concat_outputs' not in tonode2.name \
                                            and not last_is_reshape(tonode2):
                                        triple = (
                                            fromnode.op_type, tonode.op_type, tonode2.op_type)
                                        if triple not in optriples:
                                            optriples.append(triple)

    return cnt_op, len(pairs), ops, oppairs, optriples


def get_coverage(modellist, test_set, g_metrics, block_corpus):

    for i in range(len(modellist)):
        model = modellist[i]
        for model_out_i, output in enumerate(model.graph.output):
            for node in model.graph.node:
                for node_out_i, nodeout in enumerate(node.output):
                    if nodeout == output.name:
                        output.name = 'model_output_'+str(model_out_i)
                        node.output[node_out_i] = 'model_output_' + \
                            str(model_out_i)

        edges, test_set, dims, newattrcnt = parse_graph(
            model, test_set, block_corpus)
        opcnt, paircnt, optypes, opedges, optriples = get_graph_metric(
            model, edges)

        g_opcnt = opcnt
        g_type = len(optypes)
        g_edge = len(opedges)
        g_triple = len(optriples)

        def get_p_and_s(dims):
            ps = []
            for x in dims.values():
                if x not in ps:
                    ps.append(x)
            return ps

        g_s_and_p = len(get_p_and_s(dims)) + newattrcnt
        g_metrics.append((g_opcnt, g_type, g_edge, g_triple, g_s_and_p))

    total = [0., 0., 0., 0., 0., 0.]
    metricOP = {}
    for key in block_corpus:
        if key in test_set:
            metrics = get_metric_for_op(key, test_set, block_corpus)
            total = [total[i]+metrics[i] for i in range(len(metrics)-1)]
            metricOP[key] = metrics
        else:
            metricOP[key] = [0, '-', '-', '-', '-', '-', '0']

    metricI = [total[i]/len(block_corpus) for i in range(len(total))]
    return metricOP, metricI, g_metrics, test_set
