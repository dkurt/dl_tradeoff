import sys
import common  # open_model_zoo
import argparse

# Uncomment this code to generate a list of classification topologies.
# for entry in common.load_models(argparse.Namespace(config=None)):
#     if entry.task_type == 'classification':
#         print(entry.name)
# sys.exit(1)

import os
import json
import numpy as np
import csv
import time

import cv2 as cv


def evaluate(model, imagesDir, useIR, perfTest):
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

    with open(os.path.join(imagesDir, 'val.txt'), 'rt') as f:
        data = [line.split(' ') for line in f.read().strip('\n').split('\n')]

    accuracy = 0
    times = []
    for i, datasetEntry in enumerate(data):
        if perfTest and i == 10:
            break
        print("%d/%d" % (i + 1, len(data)))

        imgPath = os.path.join(imagesDir, datasetEntry[0])
        label = int(datasetEntry[1])
        img = cv.imread(imgPath)
        if img is None:
            print("Unable to read image: " + imgPath)
            continue

        if perfTest:
            if i == 0:
                net.classify(img)

            # Run without pre- and postprocessing
            start = time.time()
            net.forward()
            times.append(time.time() - start)
        else:
            classId, _ = net.classify(img)
            accuracy += classId == label

    if perfTest:
        return np.median(times[1:])  # Exclude first run
    else:
        return float(accuracy) / len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to evaluate image classification networks on ImageNet validation dataset.')
    parser.add_argument('-m', dest='model',
                        help='Name of topology to evaluate')
    parser.add_argument('-i', dest='images', required=True,
                        help='Path to ImageNet validation dataset. See http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads')
    parser.add_argument('-o', dest='output', required=False,
                        help='Output file to write measured metrics')
    parser.add_argument('-c', dest='config', default='classification.yml',
                        help='Path to a config file')
    parser.add_argument('-p', dest='perf', action='store_true',
                        help='Run performance test')
    args = parser.parse_args()

    if args.output:
        with open(args.output, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['model'] + ['performance (ralative)'] if args.perf else ['top-1 accuracy'])

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
            metric = evaluate(name, args.images, useIR, args.perf)
            if args.perf:
                performance[name] = metric
            elif args.output:
                with open(args.output, 'at') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, metric])

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
