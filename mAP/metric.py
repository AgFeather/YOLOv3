import numpy as np
import glob
import os
import json
import argparse
import sys
sys.path.append('..')

import utils


tmp_files_path = './tmp_files'
results_files_path = './results'
utils.reset_path(tmp_files_path)
utils.reset_path(results_files_path)



def load_ground_truth_files():
    """
    Load each of the ground-truth files into a temporary ".json" file.
    Create a list of all the class names present in the ground-truth (gt_classes).
    """
    ground_truth_files_list = glob.glob('ground_truth/*.txt')
    ground_truth_files_list.sort()
    gt_counter_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split('.txt', 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        assert os.path.exists('predicted/' + file_id + '.txt')
        lines_list = utils.file_lines_to_list(txt_file)

        # create ground-truth dictionary
        bounding_boxes = []
        for line in lines_list:
            class_name, class_index, left, top, right, bottom = line.split(';')
            bbox = left + ' ' + top + ' ' + right + ' ' + bottom
            bounding_boxes.append({'class_name':class_name, 'class_index':class_index, 'bbox':bbox, 'used':False})
            gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1
        with open(tmp_files_path + '/' +file_id + '_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes_name = list(gt_counter_per_class.keys())
    gt_classes_name = sorted(gt_classes_name)
    n_classes = len(gt_classes_name)

    return gt_classes_name, gt_counter_per_class, n_classes


def load_predicted_files(gt_classes_name):
    """
    Predicted
    Load each of the predicted files into a temporary ".json" file.
    """
    predicted_files_list = glob.glob('predicted/*.txt')
    predicted_files_list.sort()
    for index, class_name in enumerate(gt_classes_name):
        bounding_boxes = []
        for txt_file in predicted_files_list:
            file_id = txt_file.split('.txt', 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = utils.file_lines_to_list(txt_file)
            for line in lines:
                pred_class_name, pred_class_index, confidence, xmin, ymin, xmax, ymax = line.split(';')
                if pred_class_name == class_name:
                    bbox = xmin + " " + ymin + " " + xmax + " " + ymax
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(tmp_files_path + '/' + class_name + "_predictions.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    return predicted_files_list



def metric(iou_threshold=0.5):
    """
    calculate the AP for each class and mAP
    """
    gt_classes_name, gt_counter_per_class, n_classes = load_ground_truth_files()
    predicted_files_list = load_predicted_files(gt_classes_name)

    # calculate AP for each class
    sum_AP = 0.0
    results_file = open(results_files_path + '/results.txt', 'w')
    results_file.write("AP and precision/recall per class, iou_threshold\n")
    count_true_positives = {}
    for index, class_name in enumerate(gt_classes_name): # load each predicted class
        count_true_positives[class_name] = 0
        predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
        predictions_data = json.load(open(predictions_file))

        nd = len(predictions_data) # the number of predictions for each class
        tp = [0] * nd # true predict
        fp = [0] * nd # false predict
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction['file_id']
            gt_file = tmp_files_path + '/' + file_id + '_ground_truth.json'
            ground_truth_data = json.load(open(gt_file))

            iou_max = -1
            gt_match = -1
            pred_bbox = [float(x) for x in prediction['bbox'].split()]
            for obj in ground_truth_data:
                if obj['class_name'] == class_name:
                    gt_bbox = [float(x) for x in obj['bbox'].split()]
                    iou = utils.bbox_iou(pred_bbox, gt_bbox)
                    if iou > iou_max:
                        iou_max = iou
                        gt_match = obj

            if iou_max >= iou_threshold:
                if not bool(gt_match['used']):
                    tp[idx] = 1
                    gt_match['used'] = True
                    count_true_positives[class_name] += 1
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1

        precision, recall = cal_recall_precision(
            num_gt=gt_counter_per_class[class_name], tp=tp, fp=fp)

        ap, _, __ = voc_ap(recall, precision)
        sum_AP += ap
        text = "AP: {0:.2f}%".format(ap * 100) + ";\tclass_name:" + class_name
        print(text)
        results_file.write(text + '\n')
        # save inf p-r info or not.
        # rounded_prec = ['%.2f' % elem for elem in precision]
        # rounded_rec = ['%.2f' % elem for elem in recall]
        # results_file.write("Precision: " + str(rounded_prec) +
        #                    "\n Recall: " + str(rounded_rec) + "\n\n")

    results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP * 100)
    results_file.write(text + "\n")
    print(text)

    return mAP


def cal_recall_precision(num_gt, tp, fp):
    # calculate precision/recall
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    recall = tp[:]
    for idx, val in enumerate(tp):
        recall[idx] = float(tp[idx]) / num_gt
    precision = tp[:]
    for idx, val in enumerate(tp):
        precision[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    return precision, recall


def voc_ap(rec, prec):
    """
    Official matlab code VOC2012
    Calculate the AP given the recall and precision array
    first, compute a version of the measured precision/recall curve with
        precision monotonically decreasing
    second, compute the AP as the area under this curve by numerical integration.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    # make the precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # create a list of indexes where the recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    # calculate ap
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="iou threshold for mAP calculation")
    args = parser.parse_args()
    iou_threshold = args.iou_threshold
    mAP = metric(iou_threshold=iou_threshold)