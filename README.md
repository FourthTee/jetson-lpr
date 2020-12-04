# License Plate Recognition using OpenALPR

This project uses MobileNet SSD object detection and OpenLPR to recognize license plate. The mobilenet model was inferenced using Mxnet, Jetson Inference Package, and TVM.

## 1) Installation


## 2) Configure

In the `settings.yaml` file there are parameters to set up

- mode

    There are three options for the mode
    - `tvm`: run inference using TVM runtime
    - `mxnet`: run inference using mxnet runtime
    - `jet_inference`: run inference using jetson-inference package (only works when running on Nvidia Jetson devices)

- model_dir_file

    Only need to set when running in `tvm` mode

    Set to `path` to directory containing `[model].so`, `[model].json`, and `[model].params` file

- language

    Set to the region in which the license plates are from

    Currently options are `us`, `eu`, `au`, and `kr`

- video_camera

    Set to the `camera_number` that will capture the license plate

    ```
    /dev/video[camera_number]
    ```

- target

    Set the target for running inference

    - `llvm`: for running on CPU
    - `cuda`: for running on GPU


## 3) Run

```
python3 main.py
```

## 4) Benchmark

The performance of each configuration on https://docs.google.com/spreadsheets/d/1oMSxabF4l3xrvqK6NWnBud4ajrDd7Hn4ok-dFK1BRNw/edit?usp=sharing