# Deep learning network accuracy / efficiency tradeoff diagrams

This repository contains data and evaluation scripts for plots from
https://dkurt.github.io/dl_tradeoff which represent comparison of different
computer vision deep learning networks by accuracy and efficiency.

This kind of diagrams can help to choose pre-trained networks for your problem
or decide which topology / backbone is more suitable for training.

> **NOTE**: Explore datasets that were used for evaluation! Some of the models can be
trained for specific use cases and may perform better for some of the scenarios.
To put as much networks as possible to a single chart we had to evaluate them on
the same data to make metrics comparable. Image previews will be added later.

## How can I add a new network?

There are two sources which are used for evaluation: [Open Mozel Zoo](https://github.com/opencv/open_model_zoo)
which is more preferable and [custom models](./extra/models). Choose one of them for contribution.

## Found a bug in metric measurement or have concerns about it?

Open [an issue](https://github.com/dkurt/dl_tradeoff/issues) or contribute
changes by [a pull request](https://github.com/dkurt/dl_tradeoff/pulls).

Branches strategy:
* [master](https://github.com/dkurt/dl_tradeoff/tree/master) - release versions. Is used for rendering.
* [gh-pages](https://github.com/dkurt/dl_tradeoff/tree/gh-pages) - development branch (choose one for new pull requests).

## Local experiments

If you want to try to reproduce the data, follow these steps:

1. Clone Open Model Zoo
  ```bash
  git clone https://github.com/opencv/open_model_zoo
  git remote add dkurt https://github.com/dkurt/open_model_zoo
  git fetch dkurt py_open_model_zoo_v2
  git checkout py_open_model_zoo_v2
  export PYTHONPATH=/path/to/open_model_zoo/tools/downloader:$PYTHONPATH
  ```

2. Install OpenCV at least of version 4.1.2 or starts with OpenVINO R3

3. Download task specific dataset:
  * COCO for object detection: http://cocodataset.org/#download
  * ImageNet for classification: http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
  * FDDB for face detection: http://vis-www.cs.umass.edu/fddb/

4. (optional for object detection) Install COCO validation pipeline
  ```bash
  git clone https://github.com/cocodataset/cocoapi
  cd cocoapi/PythonAPI
  python3 setup.py build_ext --inplace
  rm -rf build
  export PYTHONPATH=/path/to/cocoapi/PythonAPI:$PYTHONPATH
  ```
