import os
import cv2
import random
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg
from core import preprocess


class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size,
                                    self.train_input_size, 3))

            # 对每个分辨率上的每个anchor都设置一个label，全0初始化
            batch_label_sbbox = np.zeros(
                (self.batch_size, self.train_output_sizes[0],
                 self.train_output_sizes[0], self.anchor_per_scale,
                 5 + self.num_classes))
            batch_label_mbbox = np.zeros(
                (self.batch_size, self.train_output_sizes[1],
                 self.train_output_sizes[1], self.anchor_per_scale,
                 5 + self.num_classes))
            batch_label_lbbox = np.zeros(
                (self.batch_size, self.train_output_sizes[2],
                 self.train_output_sizes[2], self.anchor_per_scale,
                 5 + self.num_classes))

            batch_sbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)

                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                        self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([
            list(map(lambda x: int(float(x)), box.split(',')))
            for box in line[1:]])

        if self.data_aug:
            image, bboxes = preprocess.random_horizontal_flip(
                np.copy(image), np.copy(bboxes))
            image, bboxes = preprocess.random_crop(
                np.copy(image), np.copy(bboxes))
            image, bboxes = preprocess.random_translate(
                np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image),
                                               [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def smooth_label(self, bbox_class_index):
        # 对 class label进行 smooth_onehot
        onehot = np.zeros(self.num_classes, dtype=np.float)
        onehot[bbox_class_index] = 1.0
        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
        return smooth_onehot

    def preprocess_true_boxes(self, bboxes):
        # 根据每张图片的bboxes，生成 sbbox, mbbox, lbbox对应的true label以及回归坐标
        label = [
            np.zeros((self.train_output_sizes[i], self.train_output_sizes[i],
                      self.anchor_per_scale, 5 + self.num_classes))
            for i in range(3)] #[(13,13,3,85), (26,26,3,85), (52,52,3,85)]
        #bboxes_xywh(3, 150, 4)表示一张图片中最多可以存放(3, 150)个真实框
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3, ))  # 对应3种网格尺寸的bounding box数量

        for bbox in bboxes:  # 对图片中的每个真实框处理
            bbox_coordinate = bbox[:4]  #(x_min, y_min, x_max, y_max)
            bbox_class_index = bbox[4]

            smooth_onehot = self.smooth_label(bbox_class_index)

            # 计算中心点坐标, 计算宽高, 拼接成一个数组(x, y, w, h)
            bbox_xywh = np.concatenate(
                [(bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                 bbox_coordinate[2:] - bbox_coordinate[:2]], axis=-1)

            # 按8，16，32下采样比例对中心点以及宽高进行缩放,shape = (3, 4)
            bbox_xywh_scaled = 1.0 * bbox_xywh[
                np.newaxis, :] / self.strides[:, np.newaxis]

            # 新建一个空列表，用来保存3个anchor框(先验框)和真实框(缩小后)的IOU值
            iou = []
            exist_positive = False
            for i in range(3):  # 对于一个bbox，遍历其sbbox, mbbox, lbbox，找到对应的anchors
                anchors_xywh = np.zeros((self.anchor_per_scale, 4)) # (3, 4)
                anchors_xywh[:, 0:2] = np.floor(
                    bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 找到bbox在尺度i下对应的anchors的中心坐标
                anchors_xywh[:, 2:4] = self.anchors[i] # 添加i尺度下3个anchor的预置宽高

                # 计算当前该anchor和true label的iou值
                iou_scale = utils.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :],
                                          anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask): # 三个anchor中存在最少一个和true bbox的iou大于阈值
                    # 根据真实框的坐标信息来计算所属网格左上角的位置坐标
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:  # 在该真实框中，3种网格尺寸都不存在iou > 0.3 的 anchor 框
                # reshape(-1)将矩阵排成1行，axis=-1，argmax最后返回一个最大值索引
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                # 获取best_anchor_ind所在的网格尺寸索引
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                # 获取best_anchor_ind在该网格尺寸下的索引
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] %
                               self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
