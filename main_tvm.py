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

supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc", "ssd_512_vgg16_atrous_coco",
]

alpr = Alpr("eu", "/etc/openalpr/openalpr.conf","/home/fourth/Desktop/repo/openalpr/runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
alpr.set_top_n(20)
alpr.set_default_region("md")

model_name = supported_model[3]
h = 480
w = 640
dshape = (1, 3, h, w)
block = model_zoo.get_model(model_name, pretrained=True)
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


target = "cuda"
ctx = tvm.context(target, 0)
#target = "llvm"
#ctx = tvm.cpu()
if ctx.exist:
    graph, lib, params = build()
print("Starting video stream...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")
# Set properties. Each returns === True on success (i.e. correct resolution)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

m = graph_runtime.create(graph, lib, ctx)
m.load_params(params)
fps = FPS().start()
while True:
    try:
        start = time.time()
        ret, frame = cap.read()
        #cv2.imshow("frame",frame)
        oframe = frame
        #print(frame.shape)
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        x, img = data.transforms.presets.ssd.transform_test(frame, short=480)
        #x = x.as_in_context(ctx)
        #print(x.shape)
        #start = time.time()
        class_IDs, scores, bounding_boxs = run(x, m, ctx)
        #print(bounding_boxs)
        #print(time.time() - start)
        plates = []
        confidence = []
        #print("start enumeration")
        class_IDs = class_IDs.asnumpy()
        bounding_boxs = bounding_boxs.asnumpy()
        scores = scores.asnumpy()
        end = time.time()
        #print("Runtime: "+str(end - start))
        #img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxs[0], scores[0], class_IDs[0], class_names=block.classes)
        #gcv.utils.viz.cv_plot_image(img)
        #print(net.classes)
        for i, obj in enumerate(class_IDs[0]):
                if scores[0][i][0] > 0.6:
                    if obj[0] in [5, 6]:
                        #print("Found")
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
                            plates.append(results['results'][0]['plate'])
                            confidence.append(results['results'][0]['confidence'])
                            cv2.putText(oframe, plate, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        
                else:
                    break
            
        if plates and confidence:
            print("Plates: "+str(plates))
            print("Confidence: "+str(confidence))
        cv2.imshow('frame',oframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
    except KeyboardInterrupt:
            break
    
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
"""
ax = utils.viz.plot_bbox(
    img,
    bounding_boxs.asnumpy()[0],
    scores.asnumpy()[0],
    class_IDs.asnumpy()[0],
    class_names=block.classes,
)
#print(class_IDs.asnumpy()[0])
plt.show()
"""