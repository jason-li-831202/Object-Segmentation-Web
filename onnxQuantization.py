import onnx, time
import argparse
import onnxconverter_common as m 
# from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

parser = argparse.ArgumentParser(description='https://github.com/jason-li-831202/Vehicle-CV-ADAS')
parser.add_argument('--input_onnx_model', '-i', default='./models/yolov8n-seg-coco.onnx', type=str, help='onnx model path.')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def file_size(path: str):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0
    
if __name__ == "__main__":
    args = parser.parse_args()

    input_onnx_model  = args.input_onnx_model
    assert Path(input_onnx_model).exists(), print(colorstr("red", "File=[ %s ] is not exist. Please check it !" %input_onnx_model ))
    basePath = Path(input_onnx_model).parent
    baseName = Path(input_onnx_model).stem
    baseaSuffix = Path(input_onnx_model).suffix
    output_quant_model  = str(Path.joinpath(basePath, baseName+"_fp16"+baseaSuffix))

    start = time.time()
    # Load your model.
    print(colorstr("Loading onnx path [%s]." %input_onnx_model))
    onnx_model = onnx.load(input_onnx_model)
    graph = onnx_model.graph
    nodes  = graph.node

    print(colorstr("Starting ONNX optimization export with onnx %s..." % onnx.__version__))
    # Remove the tanh operation from AnimeGAN.
    del_nodes = []
    for i, node in enumerate(nodes):
        try :
            status = list( map(lambda node_name: "Tanh" in node_name, [node.output[0], node.input[0]]) )
        except :
            statis = False
        if any(status) :
            if not del_nodes :
                old_output_node_shape = [ dim.dim_value if dim.dim_param == "" else dim.dim_param \
                                         for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
                graph.output.remove(onnx_model.graph.output[0])

                cnannel = old_output_node_shape.pop(-1)
                old_output_node_shape.insert(1, cnannel )

                new_output_node = onnx.helper.make_tensor_value_info("generator/Quant_output:0", onnx.TensorProto.FLOAT, old_output_node_shape)
                onnx_model.graph.output.extend([new_output_node])

                print(colorstr('bright_black', f"Connect layer [{nodes[i-1].output[0]}] -> [{new_output_node.name}] "))
                nodes[i-1].output[0] = new_output_node.name
            del_nodes.append(node)

    for node in del_nodes :
        print(colorstr('red', f"[Delete] Node_name : {node.name} | Op_type : {node.op_type}" ))
    [nodes.remove(node) for node in del_nodes]

    # Convert tensor float type from your input ONNX model to tensor float16.
    try:
        onnx_model = m.float16.convert_float_to_float16(onnx_model, keep_io_types=False, 
                                                    op_block_list=["Resize", 'Upsample','Reciprocal', 'ReduceMean'])
        onnx.save(onnx_model, output_quant_model)

    except Exception as e:
        print(colorstr('red', f'Eexport failure ❌ : {e}'))
        exit()

    convert_time = time.time() - start
    print(colorstr(f'\nExport complete success ✅ {convert_time:.1f}s'
                    f"\nResults saved to [{output_quant_model}]"
                    f"\nModel size:      {file_size(input_onnx_model):.1f} MB -> {file_size(output_quant_model):.1f} MB"
                    f'\nVisualize:       https://netron.app'))
    