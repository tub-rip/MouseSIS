import argparse
import sys
import os
import random
import json
from tqdm import tqdm
import yaml
import pandas as pd
import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection import SamYoloDetector
from src.tracker import XMemSort

random.seed(0)

def random_color():
    """Generate a random color based on a seed."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class Visualizer:
    def __init__(self, output_dir, save=False):
        self.color_dict = {}
        self.output_dir = output_dir
        self.save = save
        self.cnt = 0

        if self.save:
            os.makedirs(self.output_dir, exist_ok=True)

    def visualize_predictions(self, frame, predictions, instance_ids):
        """
        Visualize predictions on a frame.

        Args:
            frame (numpy.ndarray): The input frame of shape [height, width, 3].
            predictions (numpy.ndarray): The instance masks of shape [n_instances, height, width].
            instance_ids (numpy.ndarray): The instance IDs of length n_instances.

        Returns:
            numpy.ndarray: The frame with visualized predictions.
        """
        output_frame = frame.copy()

        for i in range(predictions.shape[0]):
            mask = predictions[i]
            instance_id = instance_ids[i]

            if instance_id not in self.color_dict:
                self.color_dict[instance_id] = random_color()

            color = self.color_dict[instance_id]

            colored_mask = np.zeros_like(output_frame)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]

            alpha = 0.5
            output_frame = cv2.addWeighted(output_frame, 1, colored_mask, alpha, 0)

            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            cv2.putText(output_frame, str(instance_id), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.save:
            output_path = os.path.join(self.output_dir, f'{str(self.cnt).zfill(6)}.png')
            cv2.imwrite(output_path, output_frame)

        self.cnt += 1
        return output_frame
    def visualize_frame(self, frame):
        """
        Visualize a frame when there is no prediction in this frame.

        Args:
            frame (numpy.ndarray): The input frame of shape [height, width, 3].

        Returns:
            numpy.ndarray: The frame
        """
        output_frame = frame.copy()

        if self.save:
            output_path = os.path.join(self.output_dir, f'{str(self.cnt).zfill(6)}.png')
            cv2.imwrite(output_path, output_frame)

        self.cnt += 1
        return output_frame

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        iou =  0.0
    else:
        iou = intersection / union
    return iou


def propagate_config(config):
    return config

def write_into_json_results(json_result, masks, ids, frame_idx, seq_id, instance_ids_list,num_frame):
    for mask, id in zip(masks, ids):
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode('utf-8')
        if id not in instance_ids_list:
            pred = {'video_id':seq_id, 'score': 1, 'instance_id': id, 'category_id': 1, 'segmentations': [None]*num_frame} 
            pred['segmentations'][frame_idx] = rle
            json_result.append(pred)
            instance_ids_list.append(id)
        else:
            for pred in json_result:
                if pred["instance_id"] == id:
                    pred["segmentations"][frame_idx] = rle
    print(f"instance_ids_list = {instance_ids_list}")
    return json_result, instance_ids_list

def NMS(gray_masks, gray_scores, e2vid_masks, e2vid_scores, iou_threshold):
    if e2vid_masks is None:
        e2vid_preds = []
    else:    
        e2vid_preds = [{'mask': mask, 'score': score} for mask, score in zip(e2vid_masks, e2vid_scores)]
    if gray_masks is None:
        gray_preds = []
    else:
        gray_preds = [{'mask': mask, 'score': score} for mask, score in zip(gray_masks, gray_scores)]
    
    preds = gray_preds + e2vid_preds
    preds.sort(key=lambda x: x['score'], reverse=True)
    combined_masks = []
    while preds:
        best_pred = preds.pop(0)
        combined_masks.append(best_pred)
        filtered_preds = []
        for pred in preds:
            mean_iou = calculate_iou(best_pred['mask'], pred['mask'])
            # filter all the matches lower than iou_threshold, and go into next round of matching
            if mean_iou < iou_threshold:
                filtered_preds.append(pred)
        preds = filtered_preds

    combined_masks = [pred['mask'] for pred in combined_masks]
    combined_masks = np.stack(combined_masks, axis=0)
    return combined_masks

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/predict/combined.yaml')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = propagate_config(config)

    data_root_frame = config['common']['data_root_frame']
    data_root_e2vid = config['common']['data_root_e2vid']
    data_format  = config['common']['data_format']

    gray_detector = SamYoloDetector(**config['gray_detector'])
    e2vid_detector = SamYoloDetector(**config['e2vid_detector'])
    tracker = XMemSort(**config['tracker'])

    # Track and Segment for every sequence
    dataset_info = pd.read_csv(config['common']['metadata_path'])
    testset_info = dataset_info[dataset_info['split'] == "test"]
    test_seqs = testset_info.apply(lambda row: str(row['sample_id']).zfill(2), axis=1).tolist()
    
    # avoid seq 01 and 07 for bad e2vid reconstruction
    if data_format == "e2vid" or data_format == "combined":
        test_seqs = [item for item in test_seqs if item not in ['01', '07']]

    for seq in test_seqs:  
        instance_ids = []
        json_result = [] 
        seq_dir_frame = os.path.join(data_root_frame, "test", 'images', seq)
        seq_dir_e2vid = os.path.join(data_root_e2vid, "test", 'images', seq)
        if data_format == "e2vid" or data_format == "combined":
            frame_names = sorted(os.listdir(seq_dir_e2vid))
        else:
            frame_names = sorted(os.listdir(seq_dir_frame))
        num_frame = len(frame_names)
        iou_threshold = config['NMS']['iou_threshold']
  
        output_dir = os.path.join(config['output_dir'], str(iou_threshold), seq)
        viz = Visualizer(output_dir, save=True)
        for i, frame_name in tqdm(enumerate(frame_names), total=len(frame_names), desc=seq):
            
            frame_path = os.path.join(seq_dir_frame, frame_name)
            frame = cv2.imread(frame_path)

            e2vid_frame_path = os.path.join(seq_dir_e2vid, frame_name)
            e2vid_frame = cv2.imread(e2vid_frame_path)

            if data_format == 'frame':
                gray_masks, gray_scores = gray_detector.run(frame)
                gray_masks = NMS(gray_masks, gray_scores, None, None, iou_threshold)
                if gray_masks is None:
                    viz.visualize_frame(frame)
                    continue
                active_trackers = tracker.update(gray_masks, frame)
            elif data_format == 'e2vid':
                e2vid_masks, e2vid_scores = e2vid_detector.run(e2vid_frame)
                if e2vid_masks is None:
                    viz.visualize_frame(frame)
                    print("no mask detected by e2vid detector")
                    continue
                active_trackers = tracker.update(e2vid_masks, frame)
            elif data_format == 'combined':
                gray_masks, gray_scores = gray_detector.run(frame)
                e2vid_masks, e2vid_scores = e2vid_detector.run(e2vid_frame)
                combined_masks = NMS(gray_masks, gray_scores, e2vid_masks, e2vid_scores, iou_threshold)
                if combined_masks is None:
                    viz.visualize_frame(frame)
                    continue
                active_trackers = tracker.update(combined_masks, frame)

            viz.visualize_predictions(frame, active_trackers['masks'], active_trackers['ids'])
            json_result, instance_ids= write_into_json_results(json_result, active_trackers['masks'], active_trackers['ids'], i, seq, instance_ids,num_frame)
        
        os.makedirs(output_dir, exist_ok=True)
        output_json_path = os.path.join(output_dir, "results.json")
        with open(output_json_path, 'w') as f:
            json.dump(json_result, f, indent=4)

    final_results = []
    output_folder = os.path.join(config['output_dir'], str(iou_threshold))
    for seq_folder in os.listdir(output_folder):
        seq_results_json_path = os.path.join(output_folder, seq_folder, 'results.json')
        if os.path.isfile(seq_results_json_path):
            with open(seq_results_json_path, 'r') as f:
                results = json.load(f)
                final_results.extend(results)
    
    final_results_path = os.path.join(output_folder, 'final_results.json')
    output_dir_eval = os.path.join("src/TrackEval/data/trackers", f"{data_format}_{str(iou_threshold)}","test")
    os.makedirs(output_dir_eval, exist_ok=True)
    final_results_path_eval = os.path.join(output_dir_eval, "final_results.json")

    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    with open(final_results_path_eval, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f'Merged results saved to {final_results_path} and {final_results_path_eval}')
    print('Done.')

if __name__ == '__main__':
    main()