"""yolo_with_plugins.py

Implementation of TrtYOLO class with the yolo_layer plugins.
"""


from __future__ import print_function

import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit  # This is needed for initializing CUDA driver


try:
    ctypes.cdll.LoadLibrary('/app/plugins/libyolo_layer.so')
except OSError as e:
    raise SystemExit('ERROR: failed to load ./plugins/libyolo_layer.so.  '
                     'Did you forget to do a "make" in the "./plugins/" '
                     'subdirectory?-line 20 yolo_with_plugins') from e


def _preprocess_yolo(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def _nms_boxes(detections, nms_threshold,img_w=416,img_h=416):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]*img_w
    y_coord = detections[:, 1]*img_h
    width = detections[:, 2]*img_w
    height = detections[:, 3]*img_h
    box_confidences = detections[:, 4]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        # print("iou",iou)
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep


def _postprocess_yolo(trt_outputs, img_w, img_h, conf_th, nms_threshold=0.5):
    """Postprocess TensorRT outputs.

    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold

    # Returns
        boxes, scores, classes (after NMS)
    """
    # concatenate outputs of all yolo layers
    detections = np.concatenate(
        [o.reshape(-1, 7) for o in trt_outputs], axis=0)

    # drop detections with score lower than conf_th
    box_scores = detections[:, 4] * detections[:, 6]
    pos = np.where(box_scores >= conf_th)
    detections = detections[pos]

    # scale x, y, w, h from [0, 1] to pixel values
    # detections[:, 0] *= img_w
    # detections[:, 1] *= img_h
    # detections[:, 2] *= img_w
    # detections[:, 3] *= img_h

    # NMS
    nms_detections = np.zeros((0, 7), dtype=detections.dtype)
    for class_id in set(detections[:, 5]):
        idxs = np.where(detections[:, 5] == class_id)
        cls_detections = detections[idxs]
        keep = _nms_boxes(cls_detections, nms_threshold, img_w, img_h)
        nms_detections = np.concatenate(
            [nms_detections, cls_detections[keep]], axis=0)
    
    #custom
    result = []
    # print("nms_detections",nms_detections)
    if len(nms_detections) == 0:
        # boxes = np.zeros((0, 4), dtype=np.int)
        # scores = np.zeros((0, 1), dtype=np.float32)
        # classes = np.zeros((0, 1), dtype=np.float32)
        result = []
    else:
        for detection in nms_detections:
            obj = {}
            obj["name"] = "Face"
            obj["class_id"] = int(detection[5])
            obj['prob'] = float(detection[4] * detection[6])
            # detection[0] /= img_w
            # detection[1] /= img_h
            # detection[2] /= img_w
            # detection[3] /= img_h
            obj["bbox"] = detection[:4].astype(np.float).tolist()
            result.append(obj)
        # xx = nms_detections[:, 0].reshape(-1, 1)
        # yy = nms_detections[:, 1].reshape(-1, 1)
        # ww = nms_detections[:, 2].reshape(-1, 1)
        # hh = nms_detections[:, 3].reshape(-1, 1)
        # boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        # boxes = boxes.astype(np.int)
        # scores = nms_detections[:, 4] * nms_detections[:, 6]
        # classes = nms_detections[:, 5]


    return result


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, grid_sizes):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        print(binding)
    print(engine)
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            print(size,grid_sizes[output_idx])
            assert size == grid_sizes[output_idx] * 3 * 7 * engine.max_batch_size
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """do_inference (for TensorRT 6.x or lower)

    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def get_yolo_grid_sizes(model_name, h, w):
    """Get grid sizes (w*h) for all yolo layers in the model."""
    if 'yolov3' in model_name:
        if 'tiny' in model_name:
            return [(h // 32) * (w // 32), (h // 16) * (w // 16)]
        else:
            return [(h // 32) * (w // 32), (h // 16) * (w // 16), (h // 8) * (w // 8)]
    elif 'yolov4' in model_name:
        if "3l" in model_name and 'tiny' in model_name:
            return [(h // 32) * (w // 32), (h // 16) * (w // 16), (h // 8) * (w // 8)]
        elif 'tiny' in model_name:
            return [(h // 32) * (w // 32), (h // 16) * (w // 16)]
        else:
            return [(h // 8) * (w // 8), (h // 16) * (w // 16), (w // 32) * (h // 32)]
    else:
        raise ValueError('ERROR: unknown model (%s)!' % args.model)


class TrtYOLO(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = '%s.trt' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model='yolov4-tiny-3l-416', input_shape=(416,416), category_num=1, cuda_ctx=None):

        """
        model = yolov4-tiny-3l-416
        input_shape = (416,416)
        category_num=1
        Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.category_num = category_num
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.inference_fn = do_inference if trt.__version__[0] < '7' \
                                         else do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        try:
            self.context = self.engine.create_execution_context()
            grid_sizes = get_yolo_grid_sizes(
                self.model, self.input_shape[0], self.input_shape[1])
            print(grid_sizes)
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine, grid_sizes)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, img, conf_th=0.5, nms_thresh = 0.5):

        """
        Detect objects in the input image.
        Runturn 
        bboxes : in format (x_min, y_min, x_max, y_max) 
        [[218 186 265 256]
         [  2 174  42 236]
         [451 210 483 247]
         [ 84 200 120 239]
         [597 143 673 264]
         [284 214 315 250]
         [906 185 954 246]
         [507 207 527 233]
         [775 192 809 235]
         [338 196 363 229]
         [684 187 714 228]]
        
        scroces
         [0.91539466 0.89655954 0.87355214 0.73476136 0.6624681  0.62618977
         0.58154106 0.5237454  0.4128909  0.41026872 0.30501187]
        
        class
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        """
        img_resized = _preprocess_yolo(img, self.input_shape)

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized)
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        result = _postprocess_yolo(
            trt_outputs, img.shape[1], img.shape[0], conf_th, nms_thresh)
        return result

   
