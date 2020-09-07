# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc
import argparse
import detector_pb2
import detector_pb2_grpc
import time
from PIL import Image
# from darknet.darknet import YoloDetector
from usage_tensorrt_model.yolo_with_tensorrt import TrtYOLO
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda

# import cv2
import io
import numpy as np
import json
import cv2

class ServerYoloV4(detector_pb2_grpc.FaceDetectorServicer):

    def __init__(self):
        cuda_ctx = cuda.Device(0).make_context()
        self.__yolo = TrtYOLO(model="usage_tensorrt_model/yolov4-tiny-3l-416", input_shape=(416,416), category_num=1, cuda_ctx=cuda_ctx)

    def detect(self, request, context):
        start = time.time()
        img = request.image
        image = Image.open(io.BytesIO(img))
        image = np.array(image)
        # detection = self.__yolo.detect(img=image, thresh=.2, hier_thresh=.2, nms=.5, width=416, height=416)
        detection = self.__yolo.detect(image)
        print("detection",detection)
        detection = json.dumps(detection)
        end = time.time()
        elapsed = end - start
        fps = 1 / elapsed
        print(elapsed, fps)
        return detector_pb2.ObjectInfo(objects='%s' % detection)


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    detector_pb2_grpc.add_FaceDetectorServicer_to_server(ServerYoloV4(), server)
    print("start server")
    server.add_insecure_port('[::]:'+port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add enviroment')
    # parser.add_argument('--video_path', default='./video/video3.mp4',
    #                     help='path to your input video (defaulte is "VMS.mp4")')
    parser.add_argument('--port','-p', default='50051',
                        help='input the port for service')
    args = parser.parse_args()
    logging.basicConfig()
    serve(args.port)
