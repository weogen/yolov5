import onnx
from onnxruntime.quantization import quantize_static,quantize_dynamic, QuantType

model_fp32 = '/home/tnc/PycharmProjects/yolov5/runs/train/zl_crane2/weights/best_smplf.onnx'
model_quant = '/home/tnc/PycharmProjects/yolov5/runs/train/zl_crane2/weights/best_dynmc_smplf_quant.onnx'
#quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

input_onnx_model = "/home/tnc/PycharmProjects/yolov5/runs/train/zl_crane2/weights/best_smplf.onnx"
output_onnx_model = '/home/tnc/PycharmProjects/yolov5/runs/train/zl_crane2/weights/best_smplf_fp16.onnx'

onnx_model = onnxmltools.utils.load_model(input_onnx_model)
onnx_model = convert_float_to_float16(onnx_model)
onnxmltools.utils.save_model(onnx_model, output_onnx_model)
