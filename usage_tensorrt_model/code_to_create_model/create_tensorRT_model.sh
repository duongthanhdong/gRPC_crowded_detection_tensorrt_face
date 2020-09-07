python3 ./usage_tensorrt_model/code_to_create_model/yolo_to_onnx.py -m ./usage_tensorrt_model/code_to_create_model/yolov4-tiny-3l-416 -c 1
python3 ./usage_tensorrt_model/code_to_create_model/onnx_to_tensorrt.py -m ./usage_tensorrt_model/code_to_create_model/yolov4-tiny-3l-416 -c 1
mv ./usage_tensorrt_model/code_to_create_model/yolov4-tiny-3l-416.trt ./usage_tensorrt_model/
