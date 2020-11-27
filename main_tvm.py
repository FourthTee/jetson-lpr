import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
import tvm.autotvm as autotvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
import cv2
import mxnet as mx
from openalpr import Alpr
import time
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import gluoncv as gcv
import argparse


supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc", "ssd_512_vgg16_atrous_coco",
]

def get_alpr(region):
    # load alpr model
    alpr = Alpr(region, "/etc/openalpr/openalpr.conf","/home/fourth/Desktop/repo/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    alpr.set_top_n(20)
    alpr.set_default_region("md")
    return alpr

def build():
    graph = open("model_opt.json").read()
    lib = tvm.runtime.load_module("./model_opt.so")
    params = bytearray(open("model_opt.params", "rb").read())
    return graph, lib, params

def run(input, mod, ctx):
    tvm_input = tvm.nd.array(input.asnumpy(), ctx=ctx)
    mod.set_input("data", tvm_input)
    mod.run()
    class_IDs, scores, bounding_boxs = mod.get_output(0), mod.get_output(1), mod.get_output(2)
    return class_IDs, scores, bounding_boxs

def evaluate(module, ctx, number, repeat):
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

def detect():
    alpr = get_alpr("eu")
    target = "cuda"
    ctx = tvm.context(target, 0)
    if ctx.exist:
        graph, lib, params = build()
    print("Starting video stream...")
    cap = cv2.VideoCapture(int(args.stream))
    if not cap.isOpened():
        raise Exception("Could not open video device")

    m = graph_runtime.create(graph, lib, ctx)
    m.load_params(params)
    fps = FPS().start()
    while True:
        
        ret, frame = cap.read()
        
        oframe = frame
        
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        x, img = data.transforms.presets.ssd.transform_test(frame, short=480)
    
        class_IDs, scores, bounding_boxs = run(x, m, ctx)

        class_IDs = class_IDs.asnumpy()
        bounding_boxs = bounding_boxs.asnumpy()
        scores = scores.asnumpy()
        
        for i, obj in enumerate(class_IDs[0]):
                if scores[0][i][0] > 0.6:
                    if obj[0] in [5, 6]:
                    
                        x1 = bounding_boxs[0][i][0]
                        y1 = bounding_boxs[0][i][1]
                        x2 = bounding_boxs[0][i][2]
                        y2 = bounding_boxs[0][i][3]
                        oframe = cv2.rectangle(oframe, (x1, y1), (x2, y2), (36,255,12), 2)
                        cropped = img[int(y1):int(y2), int(x1):int(x2)]
                        results = alpr.recognize_ndarray(cropped)
                        
                        if len(results['results']) == 0:
                            continue
                        else:
                            plate = results['results'][0]['plate']
                            confidence = results['results'][0]['confidence']
                            cv2.putText(oframe, plate + ": " + str(confidence), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        
                else:
                    break
    
        cv2.imshow('frame',oframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update() 
    fps.stop()
    print("Approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', help='Specify video stream')
    args = parser.parse_args()
    detect()