import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
import cv2
import mxnet as mx
#from openalpr import Alpr
import time
import numpy as np

supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc", "ssd_512_vgg16_atrous_coco",
]

model_name = supported_model[4]
dshape = (1, 3, 512, 512)
im_fname = download_testdata(
    "https://github.com/dmlc/web-data/blob/main/" + "gluoncv/detection/street_small.jpg?raw=true",
    "street_small.jpg",
    module="data",
)
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
block = model_zoo.get_model(model_name, pretrained=True)

def build(target):
    mod, param = relay.frontend.from_mxnet(block, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build_module.build(mod, target, params=param)
    return graph, lib, params

def run(lib, graph, mod, ctx):
    # Build TVM runtime
    
    m = graph_runtime.create(graph, lib, ctx)
    inp = x.asnumpy()
    #print(inp.shape)
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input("data", tvm_input)
    m.set_input(**params)
    #start = time.time()
    # execute
    m.run()
    evaluate(m, ctx, 10, 3)
    #end = time.time()
    #print("Runtime: "+str(end - start))
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs

def evaluate(module, ctx, number, repeat):
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


target = "cuda"
ctx = tvm.context(target, 0)
if ctx.exist:
    graph, lib, params = build(target)
    class_IDs, scores, bounding_boxs = run(lib, graph, params, ctx)
    #print(bounding_boxs)
arr = []
for i, obj in enumerate(class_IDs.asnumpy()[0]):
    if obj[0] in [2, 3, 5, 7]:
        if scores.asnumpy()[0][i][0] > 0.6:
            x1 = bounding_boxs.asnumpy()[0][i][0]
            y1 = bounding_boxs.asnumpy()[0][i][1]
            x2 = bounding_boxs.asnumpy()[0][i][2]
            y2 = bounding_boxs.asnumpy()[0][i][3]
            cropped = img[int(y1):int(y2), int(x1):int(x2)]

            print(cropped.shape)
            cv2.imwrite('im'+str(i)+'.jpg', cropped)

"""
            #results = alpr.recognize_ndarray(cropped)
            results = alpr.recognize_file('im'+str(i)+'.jpg')
            print(results)
            if results['results']:
                for i,plate in enumerate(results['results'][0]['candidates']):
                    if plate['confidence']>0.4:
                        arr.append(plate['plate'])
                        #print(arr)
                        if i>4:
                            break

print(arr)
"""





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