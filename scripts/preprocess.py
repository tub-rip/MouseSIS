import argparse  
import json
from evlib.processing.reconstruction import E2Vid
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import h5py
import copy

IMAGE_SHAPE = (720, 1280)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='Data path root (one-level above of the sequences)', type=str, default='data/orig')
    parser.add_argument('--data_format', help='target image format(frame or e2vid)', type=str, default='frame')
    args = parser.parse_args()
    return args

def h5py_converter(path: str, data_format, output_imgs_dir, seq_id):
    """
    Loads data from an .hdf5 file and converts it into images based on the provided format.

    Args:
        path (str): Path to the .hdf5 file to be loaded.
        data_format (str): Format of the data to be extracted, either 'frame' or 'e2vid'.
        output_imgs_dir (str): Directory where the output images will be saved.
        seq_id (str): Sequence identifier used in the output filenames.

    Returns:
        None: The function processes h5 file and saves images to the specified directory.
    """
    with h5py.File(path, 'r') as f:
        if data_format == "frame":
            print("Loading images from h5 file...")
            images = f['images'][1:]
            num_images = images.shape[0]
            print(f"saving images into {output_imgs_dir}")
            for i in tqdm(range(num_images)):
                if split == "test":
                    os.makedirs(os.path.join(output_imgs_dir, seq_id), exist_ok=True)
                    output_path = os.path.join(output_imgs_dir, seq_id,f"{str(i).zfill(10)}.png")
                else:
                    output_path = os.path.join(output_imgs_dir,f"seq{seq_id}_{str(i).zfill(8)}.png")
                cv2.imwrite(output_path, images[i])
            
        elif data_format == "e2vid":
            print("Reconstructing e2vid images from events...")
            reconstructor = E2Vid(image_shape=IMAGE_SHAPE, use_gpu=True)
            x = f["x"][0:-1]
            y = f["y"][0:-1]
            t = f["t"][0:-1]
            p = f["p"][0:-1]
            events = np.stack([y, x, t, p], axis=-1)  
            # Get event_indices for each frame
            ev_indices = f['img2event']
            print(f"saving e2vid images into {output_imgs_dir}")
            for i, (start, end) in tqdm(enumerate(zip(ev_indices[:-1], ev_indices[1:])),total=len(ev_indices) - 1):
                e2vid = reconstructor(events[start:end])
                if split == "test":
                    os.makedirs(os.path.join(output_imgs_dir, seq_id), exist_ok=True)
                    output_path = os.path.join(output_imgs_dir, seq_id,f"{str(i).zfill(10)}.png")
                else:
                    output_path = os.path.join(output_imgs_dir,f"seq{seq_id}_{str(i).zfill(8)}.png")
                cv2.imwrite(output_path, e2vid)

def convert_bbox_to_yolo(bbox, IMAGE_SHAPE):
    """ 
    convert bbox from (x_min, y_min, w, h) to yolo format (x_center_normalized, y_center_normalized, w, h)

    Args:
        bbox (list or tuple): A bounding box in (x_min, y_min, width, height) format.
        IMAGE_SHAPE (tuple): Image dimensions as (height, width).

    Returns:
        list: YOLO format [x_center_normalized, y_center_normalized, width_normalized, height_normalized].
    """
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    image_width = IMAGE_SHAPE[1]
    image_height = IMAGE_SHAPE[0]
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    return [x_center, y_center, width, height]   

def save_test_anno(annotations, data_root, data_format, seqs_info):
    """
    Filters and modifies the annotations for test sequences, adjusts the annotations and saves them as a new JSON file.

    Args:
        annotations (dict): Original annotations containing video and segmentation data.
        data_root (str): The root directory for data storage.
        data_format (str): Format of the data (e.g., 'frame' or 'e2vid').
        seqs_info (pandas.DataFrame): A DataFrame containing sequence metadata, including sample IDs and split information.

    Returns:
        None: The function modifies and saves the test annotations to the specified directory.
    """
    test_seqs_id = []
    for _, seq_info in seqs_info.iterrows():
        split = seq_info["split"]
        if split == "test":
            seq_id = str(seq_info["sample_id"]).zfill(2)
            test_seqs_id.append(seq_id)
    new_annotations = annotations.copy()
    filtered_videos = []
    for video in annotations['videos']:
        if video['id'] in test_seqs_id:
            video_copy = copy.deepcopy(video)
            video_copy['length'] -= 1 
            filtered_videos.append(video_copy)
    filtered_annotations = []
    for annotation in annotations['annotations']:
        if annotation['video_id'] in test_seqs_id:
            annotation_copy = copy.deepcopy(annotation)
            if annotation_copy['segmentations']:
                annotation_copy['segmentations'].pop(0) 
            if annotation_copy['bboxes']:
                annotation_copy['bboxes'].pop(0) 
            if annotation_copy['areas']:
                annotation_copy['areas'].pop(0) 
            filtered_annotations.append(annotation_copy)

    new_annotations['videos'] = filtered_videos
    new_annotations['annotations'] = filtered_annotations
    output_dir = os.path.join(data_root.replace('/orig', ''), "preprocessed", data_format, "test")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir,"label_ytvis.json")
    with open(output_path, 'w') as json_file:
        json.dump(new_annotations, json_file, indent=4)

if __name__ == '__main__':
    args = parse_args()
    data_root_path = args.data_root
    seqs_info_path = os.path.join(data_root_path,"dataset_info.csv")
    annotations_path = os.path.join(data_root_path,"annotation.json")

    seqs_info = pd.read_csv(seqs_info_path)
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
    # extract and save youtube vis format label of test set for evaluation
    save_test_anno(annotations, args.data_root, args.data_format, seqs_info)
    for idx, seq_info in seqs_info.iterrows():
        seq_id = str(seq_info["sample_id"]).zfill(2)
        # skip seq 01 to seq 07 for bad e2vid reconstruction
        if seq_id in ["01","02","03","04","05","06","07"] and args.data_format == "e2vid":
            continue
        print(f"processing seq {seq_id}")
        split = seq_info["split"]
        num_frames = seq_info["num_frame"]-1

        # process and save annotations
        annotations_seq =  [item for item in annotations["annotations"] if item.get("video_id") == seq_id]
        output_labels_dir = os.path.join(args.data_root.replace('/orig', ''), "preprocessed", args.data_format,split,"labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for i in range(num_frames):
            bboxes = []
            for anno in annotations_seq:
                # bbox(x_min, y_min, w, h)
                # align frame with e2vid so we ignore frame 0
                bbox = anno["bboxes"][i+1]
                if bbox is not None:
                    bbox_yolo = convert_bbox_to_yolo(bbox, IMAGE_SHAPE)
                    bboxes.append(bbox_yolo)
            output_txt_path = os.path.join(output_labels_dir,f"seq{seq_id}_{str(i).zfill(8)}.txt")
            with open(output_txt_path, 'w') as f:
                for bbox in bboxes:
                    line = f"0 {' '.join(map(str, bbox))}\n"
                    f.write(line)

        # process and save frames
        h5_path = os.path.join(data_root_path, "top", f"seq{seq_id}.h5")
        output_imgs_dir = os.path.join(args.data_root.replace('/orig', ''), "preprocessed", args.data_format,split,"images")
        os.makedirs(output_imgs_dir, exist_ok=True)
        h5py_converter(h5_path, args.data_format, output_imgs_dir, seq_id)