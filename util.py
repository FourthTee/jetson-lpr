import cv2
from openalpr import Alpr
import numpy as np
import tvm

def open_cam_usb(dev, width, height):
    """Return the videocapture object with GStreamer backend"""

    gst_str = ("v4l2src device=/dev/video{} ! "
               "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
               "videoconvert ! appsink").format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def get_alpr(region):
    """Return Alpr object use to read license plates"""

    alpr = Alpr(region, "/etc/openalpr/openalpr.conf","/home/fourth/Desktop/repo/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    alpr.set_top_n(20)
    alpr.set_default_region("md")
    return alpr

def evaluate(module, ctx, number, repeat):
    """ Comput time cost of run function and print inference time """

    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

def get_bbox(bounding_boxs, idx):
    """ Return x1, y1, x2, y2 coordinates of bbox"""

    x1 = bounding_boxs[0][idx][0]
    y1 = bounding_boxs[0][idx][1]
    x2 = bounding_boxs[0][idx][2]
    y2 = bounding_boxs[0][idx][3]
    return x1, y1, x2, y2

def convertAsNumpy(classIDs, bboxes, scores):
    """Return classID, bbox, and scores converted into numpy array format"""

    classIDs = classIDs.asnumpy()
    bboxes = bboxes.asnumpy()
    scores = scores.asnumpy()
    return classIDs, bboxes, scores

def build(dir):
    """Return TVM graph, lib, params objects given the directory of .json, .params, and .so files"""

    graph = open(dir+"/model_opt.json").read()
    lib = tvm.runtime.load_module(dir+"/model_opt.so")
    params = bytearray(open(dir+"/model_opt.params", "rb").read())
    return graph, lib, params

def run(input, mod, ctx):
    """Return the results of making inference using TVM on the given input and context"""

    tvm_input = tvm.nd.array(input.asnumpy(), ctx=ctx)
    mod.set_input("data", tvm_input)
    mod.run()
    class_IDs, scores, bounding_boxs = mod.get_output(0), mod.get_output(1), mod.get_output(2)
    return class_IDs, scores, bounding_boxs