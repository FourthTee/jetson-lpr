import cv2
from openalpr import Alpr
import numpy as np
import tvm
import sys


def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30 / 1,
    flip_method=0,
):
    """ Return GStreamer pipeline for video capture given the width and height """

    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def open_cam_usb(dev, width, height):
    """Return the videocapture object with GStreamer backend"""

    gst_str = (
        "v4l2src device=/dev/video{} ! "
        "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
        "videoconvert ! appsink"
    ).format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def get_alpr(region):
    """Return Alpr object with the specified region of license plate"""

    alpr = Alpr(
        region,
        "/etc/openalpr/openalpr.conf",
        "/home/fourth/Desktop/repo/openalpr/runtime_data",
    )
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    alpr.set_top_n(20)
    alpr.set_default_region("md")
    return alpr


def evaluate(module, ctx, number, repeat):
    """ Comput time cost of TVM's run function and print inference time """

    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )


def get_bbox(bounding_boxs, idx):
    """ Return x1, y1, x2, y2 coordinates of bbox given the object's index"""

    x1 = bounding_boxs[0][idx][0]
    y1 = bounding_boxs[0][idx][1]
    x2 = bounding_boxs[0][idx][2]
    y2 = bounding_boxs[0][idx][3]
    return x1, y1, x2, y2


def convertAsNumpy(classIDs, bboxes, scores):
    """Return classIDs, bboxs, and scores converted into numpy array format"""

    classIDs = classIDs.asnumpy()
    bboxes = bboxes.asnumpy()
    scores = scores.asnumpy()
    return classIDs, bboxes, scores


def build(dir):
    """Return TVM graph, lib, params objects given the directory with .json, .params, and .so files"""

    try:
        graph = open(dir + "/model_opt.json").read()
        lib = tvm.runtime.load_module(dir + "/model_opt.so")
        params = bytearray(open(dir + "/model_opt.params", "rb").read())
    except FileNotFoundError:
        print("model_file_dir has does not contain correct files")
        sys.exit(1)

    return graph, lib, params


def run(input, mod, ctx):
    """Return the results of making inference using TVM on the given input and context"""

    tvm_input = tvm.nd.array(input.asnumpy(), ctx=ctx)
    mod.set_input("data", tvm_input)
    mod.run()
    class_IDs, scores, bounding_boxs = (
        mod.get_output(0),
        mod.get_output(1),
        mod.get_output(2),
    )
    return class_IDs, scores, bounding_boxs


def draw_plates(class_IDs, scores, bounding_boxs, oframe, img, alpr):
    """
    Search for vehicles, read license plate, draw box with license plate and returns the frame

    Parameters
    ----------
    class_IDs:
        Array of class_IDs of objects found by model
    scores:
        Array of inference scores of objects found by model
    bounding_boxs:
        Array of bounding boxes (x1, y1, x2, y2) of objects found by mdoel
    oframe:
        Original frame from captured from camera
    img:
        Preprocessed image
    alpr:
        Alpr object used for license plate reading
    """

    for i, obj in enumerate(class_IDs[0]):
        if scores[0][i][0] > 0.6:
            if obj[0] in [5, 6]:
                x1, y1, x2, y2 = get_bbox(bounding_boxs, i)
                oframe = cv2.rectangle(oframe, (x1, y1), (x2, y2), (36, 255, 12), 2)
                cropped = img[int(y1) : int(y2), int(x1) : int(x2)]
                results = alpr.recognize_ndarray(cropped)

                if len(results["results"]) == 0:
                    continue
                else:
                    plate = results["results"][0]["plate"]
                    confidence = results["results"][0]["confidence"]
                    cv2.putText(
                        oframe,
                        plate + ": " + str(confidence),
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )
        else:
            break
    return oframe