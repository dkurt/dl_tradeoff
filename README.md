# Deep learning network accuracy / efficiency tradeoff diagrams

TODO:
- [ ] Provide code to get accuracy and efficiency metrics
- [ ] Reorder COCO classes so the most interesting will go first: "person", "cat", etc.
- [ ] Size of scatter points depends on model size

## How to add a new network?

Set of models is based on [Open Mozel Zoo](https://github.com/opencv/open_model_zoo).


```bash
git clone https://github.com/opencv/open_model_zoo
git remote add dkurt https://github.com/dkurt/open_model_zoo
git fetch dkurt py_open_model_zoo_v2
git checkout py_open_model_zoo_v2
```

```bash
export PYTHONPATH=/path/to/open_model_zoo/tools/downloader:$PYTHONPATH
export PYTHONPATH=/path/to/cocoapi/PythonAPI:$PYTHONPATH
```

* Install COCO validation pipeline
  ```bash
  git clone https://github.com/cocodataset/cocoapi
  cd cocoapi/PythonAPI
  python3 setup.py build_ext --inplace
  rm -rf build
  ```
