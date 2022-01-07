from prep_test_data import *
from attr_maps_methods import *
from pathlib import Path
import json
import torch
import numpy as np
import csv
import sys
import argparse
import random

"""
ROI functions
"""


def read_rois_file_as_dict(rois_filename, test_data_path):
    dict_per_row = {}
    with open(rois_filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for i, row in enumerate(reader):
            path = str(
                test_data_path / row["label"] / f"{row['File']}_{row['Patient']}"
            )
            dict_per_row[path] = row
    return dict_per_row


def get_roi(sample_path, rois_dict):
    stem_split = sample_path.stem.split("_")
    roi = None
    if f"{stem_split[0]}_{stem_split[1]}" != "0_0":
        try:
            roi = rois_dict[str(sample_path.parents[1] / "normal" / sample_path.stem)]
        except KeyError:
            roi = rois_dict[str(sample_path.parents[1] / "abnormal" / sample_path.stem)]
    return roi


def get_roi_points(roi):
    return (int(roi["left"]), int(roi["top"])), (int(roi["right"]), int(roi["bottom"]))


def transforming_roi_points(left_top, right_bottom):
    """
    parameters: rois extreme points
    """
    crop_x, crop_y = 750, 125  # each side
    y_ratio, x_ratio = 224 / 200, 224 / 1500
    top_left_cropped_resized = (
        int((left_top[0] - crop_x) * x_ratio),
        int((left_top[1] - crop_y) * y_ratio),
    )
    delta_x_resized = (right_bottom[0] - left_top[0]) * x_ratio
    delta_y_resized = (right_bottom[1] - left_top[1]) * y_ratio
    right_cropped_resized = (
        int(top_left_cropped_resized[0] + delta_x_resized),
        int(top_left_cropped_resized[1] + delta_y_resized),
    )
    return top_left_cropped_resized, right_cropped_resized


"""
Metrics definition
"""


def metric1(attr_map, top_left, right_bottom):
    roi_sum = np.sum(
        attr_map[
            max(0, top_left[1]) : max(0, right_bottom[1]),
            max(0, top_left[0]) : max(0, right_bottom[0]),
        ]
    )
    map_sum = np.sum(attr_map)

    if (
        map_sum == 0
        or top_left[0] >= attr_map.shape[0]
        or top_left[1] >= attr_map.shape[1]
        or top_left[1] == right_bottom[1]
        or top_left[0] == right_bottom[0]
    ):
        return np.nan
    return roi_sum / map_sum


"""
Attribution maps
"""


def get_maps(map_type, model, inputs):
    if map_type == "saliency_map":
        return batch_saliency(model, inputs)
    elif map_type == "grad_cam_map":
        return grad_cam_batch(model, inputs)
    elif map_type == "gb_grad_cam_map":
        return grad_cam_batch(model, inputs, gb_cam=True)


def prepare_attr_maps(map_type, attr_map, index, gb_model=None, x=None, labels=None):
    if map_type == "saliency_map":
        return prepare_saliency(attr_map, index)
    elif map_type == "grad_cam_map":
        return preparing_grad_cam(attr_map, index)
    elif map_type == "gb_grad_cam_map":
        return preparing_gb_grad_cam(attr_map, index, gb_model, x, labels)


def imshow(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_maps(map, index, x, input_filename, save, label, pred_res, map_type):
    if map_type == "saliency_map":
        saving_saliency_map(map, index, x, input_filename, save, label, pred_res)
    elif map_type == "grad_cam_map":
        saving_grad_cam_map(map, index, x, input_filename, save, label, pred_res)
    elif map_type == "gb_grad_cam_map":
        saving_gb_grad_cam(map, input_filename, save, label, pred_res, x, index)


"""
Computing metrics
"""


def compute_metrics(model, data, batch_size, rois_dict, map_type, save):
    if map_type == "gb_grad_cam_map":
        gb_model = GuidedBackpropReLUModel(model=model)
    classes = data["test"].dataset.classes
    metric_values = []
    pred_verification = []
    labels_list = []

    for i, (inputs, labels) in enumerate(data["test"]):
        sys.stdout.write("\r" + f"batch nr: {i+1}")
        if map_type != "gb_grad_cam_map":
            attr_map, score_max_index = get_maps(map_type, model, inputs)
        else:
            attr_map, score_max_index, x = get_maps(map_type, model, inputs)

        flag = False

        for index in range(len(attr_map)):
            input_filename = Path(
                data["test"].dataset.samples[i * len(attr_map) + index][0]
            ).stem

            if map_type == "gb_grad_cam_map":
                map_prepared = prepare_attr_maps(
                    map_type, attr_map, index, gb_model, inputs, labels
                )
            else:
                map_prepared = prepare_attr_maps(map_type, attr_map, index)

            label = classes[labels[index]]

            sample_path = Path(data["test"].dataset.samples[i * batch_size + index][0])

            roi = get_roi(sample_path, rois_dict)

            if roi:
                top_left, bottom_right = get_roi_points(roi)
                top_left, bottom_right = transforming_roi_points(top_left, bottom_right)
                metric_values.append(str(metric1(map_prepared, top_left, bottom_right)))

                true = labels[index]
                pred = score_max_index[index]
                if pred != true:
                    pred_res = "wrong"
                else:
                    pred_res = "ok"

                labels_list.append(label)
                pred_verification.append(pred_res)

                save_maps(
                    map_prepared,
                    index,
                    inputs,
                    input_filename,
                    save,
                    label,
                    pred_res,
                    map_type,
                )

    return metric_values, pred_verification, labels_list


def metrics_one_heartbeat(
    data_path, models_main_path, model_name, beat, batches, rois, map_type, save
):
    data_prep = DataPreparation(str(data_path))
    data, size = data_prep.create_dataloaders(batches, False, 4)
    model_path = models_main_path / f"label_{beat}/{model_name}.pth"
    print(model_path)
    model = torch.load(model_path, map_location=torch.device(0))
    model.eval()
    return compute_metrics(model, data, batches, rois, map_type, save)


def save_results(metric_values, predictions_verification, labels_true, beat, map_type):
    res_dict = {
        "values": metric_values,
        "pred_results": predictions_verification,
        "true_labels": labels_true,
    }
    with open(f"{beat}_{map_type}_metrics.json", "w") as f:
        json.dump(res_dict, f)
    f.close()


def beat_int(beat):
    d = {"final": 5, "mid": 3, "initial": 0}
    return d[beat]


def get_model_name(beat):
    d = {
        "final": "resnet50_d_10_t_18_02",
        "initial": "resnet50_d_10_t_12_10",
        "mid": "resnet50_d_09_t_15_34",
    }
    return d[beat]


def labels():
    _ = {
        "abnormal": ["A", "a", "J", "S", "V", "E", "F"],
        "normal": ["N", "L", "R", "e", "j"],
    }
    return _


def create_maps_folders():
    for attr_map_type in ["saliency_map", "grad_cam_map", "gb_grad_cam_map"]:
        for beat in ["initial", "final"]:
            folder = Path(f"../attribution_maps/{attr_map_type}") / f"label_{beat}_beat/"
            for label in ["abnormal", "normal"]:
                Path(folder / label).mkdir(parents=True, exist_ok=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-save")
    args = parser.parse_args()
    save = str(args.save)

    MODELS_PATH = Path(f"./models/")

    BATCH_SIZE = 16

    if save == "y":
        create_maps_folders()
    for attr_map_type in ["saliency_map", "grad_cam_map", "gb_grad_cam_map"]:
        print(f"\nMAP:{attr_map_type}")

        for HEARTBEAT in ["initial", "final"]:
            print(f"\nBEAT: {HEARTBEAT}\n")
            TEST_DATA_PATH = Path(f".")
            roi_file_path = list((Path.cwd() / "ROI").glob(f"{beat_int(HEARTBEAT)}_ROI.txt"))[0]

            MODEL_NAME = get_model_name(HEARTBEAT)

            if save == "y":
                folder = (
                    Path(f"./attribution_maps/{attr_map_type}")
                    / f"label_{HEARTBEAT}_beat/"
                )
                print(folder)
            else:
                folder = None

            rois_dict = read_rois_file_as_dict(roi_file_path, TEST_DATA_PATH)
            values, prediction_results, labels = metrics_one_heartbeat(
                TEST_DATA_PATH,
                MODELS_PATH,
                MODEL_NAME,
                HEARTBEAT,
                BATCH_SIZE,
                rois_dict,
                attr_map_type,
                folder,
            )
            # save_results(values, prediction_results, labels, HEARTBEAT, attr_map_type)
