from util import *
import mxnet as mx
from gluoncv import model_zoo, data, utils
import cv2

def mxnet_inf_test(frame):
    
    ctx = mx.cpu()
    model_name = "ssd_512_mobilenet1.0_voc"
    net = model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()

    x, img = data.transforms.presets.ssd.load_test(frame, short=480)
    x = x.as_in_context(ctx)

    class_IDs, scores, bounding_boxs = net(x)

    class_IDs, bounding_boxs, scores = convertAsNumpy(
        class_IDs, bounding_boxs, scores
    )

if __name__ == "__main__":
    
    im_frame = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')

    mxnet_inf_test(im_frame)
    print('Test Pass')

