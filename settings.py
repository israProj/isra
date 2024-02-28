import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--minnode', type=int,  help="MIN_NODE", default=1)
parser.add_argument('--maxnode', type=int,  help="MAX_NODE", default=200)
parser.add_argument('--pickrate', type=float,
                    help="pickExistRate", default=0.97)
parser.add_argument('--seed', type=int, help='res_file', default=123)
parser.add_argument('--case', type=str, help='pickrate/baseline', default='baseline')
parser.add_argument('--iter', type=int, help="cur iter No.",default=1)
parser.add_argument('--times', type=int, help="times of run iteration",default=10000)
parser.add_argument('--loop', type=int, help="test round",default=1)
args = parser.parse_args()

ITER_NUM = args.times
SEED = args.seed+args.loop-1
# SEED = hash(args.file) % 10000

MIN_NODE = args.minnode
MAX_NODE = args.maxnode

pickExistRate = args.pickrate

DEBUG = False
NO_INPUT = True
TEST_TVM = True
TEST_ONNX2MLIR = False
RUN_ONNXRT = True
global_tensor_num = 0
TEST_MUFFIN = False
TEST_NNSMITH = False

MIN_TENSOR_DIM = 1
MAX_TENSOR_DIM = 5
MAX_TENSOR_DIM_LEN = 5
MAX_MULTI_INPUTS = 5
MAX_MULTI_OUTPUTS = 5

id_ops = ['Identity', 'Abs', 'Neg', 'Reciprocal', 'Floor', 'Ceil', 'Softsign',
          'Sigmoid', 'HardSigmoid', 'Relu', 'LeakyRelu', 'Elu', 'ThresholdedRelu',
          'Sin', 'Cos', 'Tanh',
          'Transpose', 'Softplus',
          'Softmax', 'MaxPool', 'AveragePool',
          'LpPool',
          'SpaceToDepth', 'Erf', 'Sign',
          'Flatten',
          'Round',
          'Exp', 'Selu', 'Sqrt',
          ]

extra_t_ops = ['MatMul', 'Add', 'Sub', 'Mul', 'Div', 'Concat']
extra_t_ops += ['Pad']  
extra_t_ops += ['BatchNormalization']
extra_t_ops += ['Expand']
extra_t_ops += ['PRelu']
extra_t_ops += ['Gemm']  
extra_t_ops += ['Conv']
# extra_t_ops += ['ConvTranspose'] #ORT contains bug

multi_extra_t_ops = ['Sum', 'Max', 'Min', 'Mean']

extra_t_ops += multi_extra_t_ops

reduce_ops = ["ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSumSquare",
              "ReduceL1", "ReduceL2", "ReduceLogSumExp", ]

multi_out_ops = ['Split']  # TVM unsupported

other_ops = []

ops = []
ops += id_ops
ops += extra_t_ops
ops += reduce_ops
ops += other_ops
ops += multi_out_ops
ops += ['Resize', 'Reshape', 'Unsqueeze', 'Slice', 'Tile', 'Gather']
ops += ['GlobalAveragePool', 'GlobalMaxPool']
ops += ['Compress']  # TVMv0.7 not support
ops += ['ReduceSum']

tvm_unsupported_ops = ['Softplus', 'Compress']  # v0.7

o2m_unsupported_ops = ['LpPool', 'ThresholdedRelu']

if TEST_TVM:
    ops = list(set(ops) - set(tvm_unsupported_ops))

if TEST_ONNX2MLIR:
    ops = list(set(ops) - set(o2m_unsupported_ops))

if TEST_MUFFIN:
    ops = json.load(open('./muffinops.json', 'r')).keys()

if TEST_NNSMITH:
    MAX_TENSOR_DIM_LEN = 16
    ops = json.load(open('./nnsmithops.json', 'r')).keys()

ops = sorted(ops)

def get_filter(get_op):
    def filter_f(x): return True
    if get_op == 'SpaceToDepth':
        def filter_f(x): return x == 4
    if get_op in ['MaxPool', 'AveragePool', 'LpPool', 'GlobalMaxPool', 'GlobalAveragePool']:
        def filter_f(x): return (x >= 3) and (
            x <= 5)  # unsupprt on ONNXRuntime/TVM
    if get_op == 'Conv':
        # TVM only supports with 1d, 2d, 3d kernel.
        def filter_f(x): return (x >= 3) and (x <= 5)
    if get_op == 'ConvTranspose':
        # TVM only supports with 1d, 2d, 3d kernel.
        def filter_f(x): return (x >= 3) and (x <= 5)
        # filter_f = lambda x: x == 4
    if get_op in ['MatMul']:
        def filter_f(x): return x >= 2
    if get_op == 'Gemm':
        def filter_f(x): return x == 2
    if get_op == 'BatchNormalization':
        def filter_f(x): return (x >= 3) and (
            x <= 5)  # unsupprt on ONNXRuntime/TVM
    if get_op == 'Unsqueeze':
        def filter_f(x): return x < MAX_TENSOR_DIM
    if get_op == 'Resize':
        def filter_f(x): return (x >= 3) and (
            x <= 5)  # unsupprt on ONNXRuntime/TVM
    if get_op in ['Elu', 'Softplus', 'ConstantOfShape']:
        def filter_f(x): return x == 1
    return filter_f


FILTER_DICT = {}
for op in ops:
    FILTER_DICT[op] = get_filter(op)
