# License Plate Recognition using OpenALPR

This project uses MobileNet SSD object detection and OpenLPR to recognize license plate. The mobilenet model was inferenced using Mxnet, Jetson Inference Package, and TVM.

## Mxnet
Make sure mxnet is installed on host machine

For GPU inference; install mxnet-cuda

To Run
```
python3 main_mxnet.py [--images] [image_dir] [--stream] [video_camera] [--visualize] [yes/no]
```

- **images**: the directory with test images
- **stream**: video camera number
- **visualize**: visualize image with bounding box (only apply when images is specified)

## Jetson Inference Package
Need to install build jetson inference package on Jetson (refer https://github.com/dusty-nv/jetson-inference)

To Run 
```
python3 main_jet_inference.py
```

## TVM
Make sure TVM is built on the host machine 

To Run
```
python3 main_tvm.py [--stream] [Video_Camera]
```
- **stream**: video camera number


## Benchmark

The performance of each configuration on https://docs.google.com/spreadsheets/d/1oMSxabF4l3xrvqK6NWnBud4ajrDd7Hn4ok-dFK1BRNw/edit?usp=sharing