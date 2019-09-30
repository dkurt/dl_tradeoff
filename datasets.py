import sys
import os
import json
import cv2 as cv
from math import pi
from fnmatch import fnmatch


# Methods which are used for datesets conversion
def addImage(dataset, imagePath):
    assert('images' in  dataset)
    imageId = len(dataset['images'])
    dataset['images'].append({
        'id': int(imageId),
        'file_name': imagePath
    })
    return imageId


def addBBox(dataset, imageId, left, top, width, height):
    assert('annotations' in  dataset)
    dataset['annotations'].append({
        'id': len(dataset['annotations']),
        'image_id': int(imageId),
        'category_id': 0,  # Face
        'bbox': [int(left), int(top), int(width), int(height)],
        'iscrowd': 0,
        'area': float(width * height)
    })


class COCODataset(object):
    def __init__(self):
        self.COCO_classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
        # replacements:
        # tvmonitor = tv
        # diningtable = dining table
        # sofa = couch
        # pottedplant = potted plant
        # motorbike = motorcycle
        # aeroplane = airplane
        self.VOC_classes = [
            'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
            'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']

        self.COCO_80_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        ]

        for dataset in [self.VOC_classes, self.COCO_80_classes]:
            for className in dataset:
                if not className in self.COCO_classes:
                    print('Class name ' + className + ' not found in COCO')
                    sys.exit(1)


    # Cast predicted class id for baseDataset to COCO classes ids
    def castClassId(self, classId, baseDataset):
        if baseDataset == 'VOC':
            return self.COCO_classes.index(self.VOC_classes[classId])
        elif baseDataset == 'COCO_80':
            return self.COCO_classes.index(self.COCO_80_classes[classId])
        elif baseDataset == 'COCO':
            return classId
        else:
            return self.COCO_classes.index(baseDataset.split(',')[classId])


    # Get class id by name
    def getClassId(self, className):
        return self.COCO_classes.index(className)


    def getAnnotations(self, annotations, images):
        return annotations


class FDDBDataset(object):
    def __init__(self):
        self.faceClassId = 0
        self.backgroundClassId = 1


    # Cast predicted class id for baseDataset to FDDB classes ids
    def castClassId(self, classId, baseDataset):
        classes = baseDataset.split(',')
        return self.faceClassId if classes[classId] == 'face' else self.backgroundClassId


    # Get class id by name
    def getClassId(self, className):
        return ['face', 'background'].index(className)


    def getAnnotations(self, annotations, images):
        dataset = {}
        dataset['images'] = []
        dataset['categories'] = [{ 'id': self.faceClassId, 'name': 'face' }]
        dataset['annotations'] = []


        def ellipse2Rect(params):
            rad_x = params[0]
            rad_y = params[1]
            angle = params[2] * 180.0 / pi
            center_x = params[3]
            center_y = params[4]
            pts = cv.ellipse2Poly((int(center_x), int(center_y)), (int(rad_x), int(rad_y)),
                                  int(angle), 0, 360, 10)
            rect = cv.boundingRect(pts)
            left = rect[0]
            top = rect[1]
            right = rect[0] + rect[2]
            bottom = rect[1] + rect[3]
            return left, top, right, bottom


        for d in os.listdir(annotations):
            if fnmatch(d, 'FDDB-fold-*-ellipseList.txt'):
                with open(os.path.join(annotations, d), 'rt') as f:
                    lines = [line.rstrip('\n') for line in f]
                    lineId = 0
                    while lineId < len(lines):
                        # Image
                        imgPath = lines[lineId]
                        lineId += 1
                        imageId = addImage(dataset, imgPath + '.jpg')

                        # Change to True to visualize annotations
                        debug = False
                        if debug:
                            img = cv.imread(os.path.join(images, imgPath) + '.jpg')

                        # Faces
                        numFaces = int(lines[lineId])
                        lineId += 1
                        for i in range(numFaces):
                            params = [float(v) for v in lines[lineId].split()]
                            lineId += 1
                            left, top, right, bottom = ellipse2Rect(params)
                            addBBox(dataset, imageId, left, top,
                                    width=right - left + 1,
                                    height=bottom - top + 1)
                            if debug:
                                cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

                        if debug:
                            cv.imshow('FDDB', img)
                            cv.waitKey()


        with open('annotations.json', 'wt') as f:
            json.dump(dataset, f)

        return 'annotations.json'
