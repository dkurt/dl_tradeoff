import os
import json
import numpy as np
import argparse
import sys
import csv
import time

import cv2 as cv
import common  # open_model_zoo

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets import *


def evaluate(model, dataset, annotations, datasetName, classes, useIR, perfTest):
    net = None
    # We do fake script location to nagivate to extra models from this repo
    fake_loc = os.path.join(os.path.dirname(__file__), 'extra', 'models', 'yolo-v3', 'model.yml')
    for path in [fake_loc, common.__file__]:
        common.__file__ = path
        for entry in common.load_models(argparse.Namespace(config=None)):
            if entry.name == model:
                # Dynamically import model class.
                t = getattr(__import__('topologies'), model.replace('-', '_').replace('.', '_'))()
                net = t.getOCVModel(useIR)  # Get OpenCV model instance (has postprocessing inside).
                sys.modules.pop('topologies')
                break
        if net:
            break

    if not net:
        print('Topology ' + model + ' not found!')
        sys.exit(1)


    if net.getLayer(0).outputNameToIndex('image_info') != -1:
        _, input_shapes, _  = net.getLayersShapes(netInputShape=None)
        inpH, inpW = int(input_shapes[0][1][2]), int(input_shapes[0][1][3])

        net.setInput(np.array([[inpH, inpW, 1.0]], dtype=np.float32), 'image_info')


    # NMS for YOLOv3
    nmsThreshold = 0.4 if len(net.getUnconnectedOutLayers()) > 1 else 0.0
    confThreshold = 0.01

    detections = []
    images = os.listdir(dataset)
    for i, imgName in enumerate(images):
        if perfTest and i == 10:
            break
        print("%d/%d" % (i + 1, len(images)))

        img = cv.imread(os.path.join(dataset, imgName))
        if img is None:
            print("Unable to read image: " + imgName)
            continue

        # TODO: fix Faster R-CNN with swapped inputs
        if net.getLayer(0).outputNameToIndex('image_info') != -1:
            net.setInput(cv.dnn.blobFromImage(img, size=(inpW, inpH)), 'image_tensor')

            start = time.time()
            out = net.forward().reshape(-1, 7)
            if perfTest:
                detections.append(time.time() - start)
                continue

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
            if perfTest:
                if i == 0:
                    net.predict(img)

                # Run without postprocessing
                start = time.time()
                net.forward()
                detections.append(time.time() - start)
                continue
            else:
                classIds, confidences, boxes = net.detect(img, confThreshold=confThreshold,
                                                          nmsThreshold=nmsThreshold)

        for classId, score, box in zip(classIds, confidences, boxes):
            classId = int(classId)
            if datasetName == 'VOC':
                categoryId = toCOCO(classId, VOC_classes)
            elif datasetName == 'COCO_80':
                categoryId = toCOCO(classId, COCO_80_classes)
            elif datasetName == 'COCO':
                categoryId = classId
            else:
                categoryId = toCOCO(classId, datasetName.split(','))

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

    if perfTest:
        return np.median(detections[1:])  # Exclude first run

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
    parser.add_argument('-m', dest='model',
                        help='Name of topology to evaluate')
    parser.add_argument('-i', dest='images', required=True,
                        help='Path to validation images folder. See http://cocodataset.org/#download')
    parser.add_argument('-a', dest='annotations', required=True,
                        help='Path to annotations folder. See http://cocodataset.org/#download')
    parser.add_argument('-o', dest='output', required=False,
                        help='Output file to write measured metrics')
    parser.add_argument('-c', dest='config', default='detection.yml',
                        help='Path to a config file')
    parser.add_argument('-p', dest='perf', action='store_true',
                        help='Run performance test')
    parser.add_argument('--classes', nargs='+', type=str, required=True,
                        help='A list of classes to be evaluated')
    args = parser.parse_args()

    if args.output:
        with open(args.output, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['model'] + ['performance (ralative)'] if args.perf else args.classes)

    fs = cv.FileStorage(args.config, cv.FILE_STORAGE_READ)
    topologies = fs.getNode('topologies')
    performance = {}
    for i in range(topologies.size()):
        entry = topologies.at(i)
        name = entry.getNode('name').string()
        datasetName = entry.getNode('dataset').string()
        useIR = entry.getNode('use_ir').string()
        useIR = not useIR or useIR.lower() == 'true'

        if not args.model or name == args.model:
            metrics = evaluate(name, args.images, args.annotations, datasetName,
                               args.classes, useIR, args.perf)
            if args.perf:
                performance[name] = metrics
            elif args.output:
                with open(args.output, 'at') as f:
                    writer = csv.writer(f)
                    writer.writerow([name] + [metrics[s] for s in args.classes])

    if args.perf:
        if args.output:
            with open(args.output, 'at') as f:
                writer = csv.writer(f)
                minTime = np.min([v for v in performance.values()])
                maxTime = np.max([v for v in performance.values()])
                for name, perf in performance.items():
                    writer.writerow([name] + [1.0 - (perf - minTime) / (maxTime - minTime)])
        else:
            print(performance)
