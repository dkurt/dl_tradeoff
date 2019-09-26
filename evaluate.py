import os
import json
import numpy as np
import argparse
import sys
import csv
from importlib import import_module

import cv2 as cv
import common  # open_model_zoo

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets import *


def evaluate(model, dataset, annotations, datasetName, classes):
    assert(datasetName in ['COCO', 'VOC', 'COCO_80'])

    topology = None
    for entry in common.load_models(argparse.Namespace(config=None)):
        if entry.name == model:
            topology = entry
            break

    if not topology:
        print('Topology ' + model + ' not found!')
        sys.exit(1)

    # Dynamically import model class.
    t = getattr(__import__('topologies'), topology.name.replace('-', '_').replace('.', '_'))()
    net = t.getOCVModel()  # Get OpenCV model instance (has postprocessing inside).

    # Looks like a bug in OpenVINO:
    # "Cannot parse parameter negative_slope from IR for ReLU."
    # "Value 0,000000 cannot be casted to float"
    if net.getLayer(0).outputNameToIndex('image_info') != -1:
        xmlPath, _ = t.getIR()

        with open(xmlPath, 'rt') as f:
            data = f.read()
        data = data.replace('type="ReLU">', 'type="ReLU"><data negative_slope="0"/>')
        with open(xmlPath, 'wt') as f:
            f.write(data)

        net = t.getOCVModel()

    # NMS for YOLOv3
    nmsThreshold = 0.4 if len(net.getUnconnectedOutLayers()) > 1 else 0.0
    confThreshold = 0.01

    detections = []
    images = os.listdir(dataset)
    for i, imgName in enumerate(images):
        print("%d/%d" % (i + 1, len(images)))

        img = cv.imread(os.path.join(dataset, imgName))
        if img is None:
            print("Unable to read image: " + imgName)
            continue

        # TODO: fix Faster R-CNN with swapped inputs
        if net.getLayer(0).outputNameToIndex('image_info') != -1:
            tensor_id = net.getLayer(0).outputNameToIndex('image_tensor')
            assert(tensor_id != -1)

            _, input_shapes, _  = net.getLayersShapes(netInputShape=None)
            inpH, inpW = int(input_shapes[0][1][2]), int(input_shapes[0][1][3])

            net.setInput(np.array([[inpH, inpW, 1.0]], dtype=np.float32), 'image_info')
            net.setInput(cv.dnn.blobFromImage(img, size=(inpW, inpH)), 'image_tensor')
            out = net.forward().reshape(-1, 7)

            classIds = []
            confidences = []
            boxes = []
            for detection in out:
                confidence = float(detection[2])
                if confidence < confThreshold:
                    continue
                classIds.append(int(detection[1]))
                confidences.append(confidence)
                x = int(detection[3] * img.shape[1])
                y = int(detection[4] * img.shape[0])
                w = int(detection[5] * img.shape[1]) - x + 1
                h = int(detection[6] * img.shape[0]) - y + 1
                boxes.append([x, y, w, h])
        else:
            classIds, confidences, boxes = net.detect(img, confThreshold=confThreshold,
                                                      nmsThreshold=nmsThreshold)

        for classId, score, box in zip(classIds, confidences, boxes):
            classId = int(classId)
            if datasetName == 'VOC':
                categoryId = toCOCO(classId, VOC_classes)
            elif datasetName == 'COCO_80':
                categoryId = toCOCO(classId, COCO_80_classes)
            else:
                categoryId = classId

            detections.append({
              "image_id": int(imgName.rstrip('0')[:imgName.rfind('.')]),
              "category_id": categoryId,
              "bbox": [int(v) for v in box],
              "score": float(score)
            })

        #     # Uncomment to render detections
        #     cv.rectangle(img, tuple(box), (0, 255, 0))
        #
        # # Uncomment to show detections
        # cv.imshow('detections', img)
        # cv.waitKey()


    resFile = 'results.json'
    with open(resFile, 'wt') as f:
        json.dump(detections, f)

    #initialize COCO ground truth api
    cocoGt=COCO(os.path.join(annotations, 'instances_val2017.json'))

    #initialize COCO detections api
    cocoDt=cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    print()

    metrics = {}
    for className in classes:
        if not className in COCO_classes:
            print('Class name not found: ' + className)
            sys.exit(1)

        print('======== %s ========' % className)
        cocoEval.params.catIds = COCO_classes.index(className)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print()
        metrics[className] = cocoEval.stats[0]
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to evaluate object detection networks on COCO 2017 validation dataset.')
    parser.add_argument('-m', dest='model', required=True,
                        help='Name of topology to evaluate')
    parser.add_argument('-i', dest='images', required=True,
                        help='Path to validation images folder. See http://cocodataset.org/#download')
    parser.add_argument('-a', dest='annotations', required=True,
                        help='Path to annotations folder. See http://cocodataset.org/#download')
    parser.add_argument('-o', dest='output', required=False,
                        help='Output file to write measured metrics')
    parser.add_argument('-c', dest='config', default='detection.yml',
                        help='Path to a config file')
    parser.add_argument('--classes', nargs='+', type=str, required=True,
                        help='A list of classes to be evaluated')
    args = parser.parse_args()

    fs = cv.FileStorage(args.config, cv.FILE_STORAGE_READ)
    topologies = fs.getNode('topologies')
    for i in range(topologies.size()):
        entry = topologies.at(i)
        name = entry.getNode('name').string()
        datasetName = entry.getNode('dataset').string()

        if name == args.model:
            metrics = evaluate(args.model, args.images, args.annotations, datasetName, args.classes)
            if args.output:
                with open(args.output, 'wt') as f:
                    writer = csv.writer(f)
                    writer.writerow(['model'] + args.classes)
                    writer.writerow([args.model] + [metrics[s] for s in args.classes])
            sys.exit(0)

    print('Unable to find ' + args.model)
    sys.exit(1)
