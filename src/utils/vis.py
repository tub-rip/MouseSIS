import cv2
import numpy as np
from skimage.measure import label, regionprops


def get_area(mask):
    assert np.all(np.isin(mask, [0, 1])), "Binary mask contains values other than 0 and 1"
    return np.sum(mask)


def get_bboxes(mask):
    labels = label(mask)
    regions = regionprops(labels)
    bboxes = [region.bbox for region in regions] # [(min_row, min_col, max_row, max_col)]
    # convert to [x_min, y_min, width, height]
    bboxes = [[bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]] for bbox in bboxes]
    return bboxes


def merge_bboxes(bboxes):
    """
    Merge multiple bounding boxes into one.

    Parameters:
    bboxes (list): List of bounding boxes. Each bounding box is in format [x_min, y_min, width, height].

    Returns:
    list: Merged bounding box in format [x_min, y_min, width, height].
    """
    x_min = min(bbox[0] for bbox in bboxes)
    y_min = min(bbox[1] for bbox in bboxes)
    x_max = max(bbox[0] + bbox[2] for bbox in bboxes)  # x_min + width
    y_max = max(bbox[1] + bbox[3] for bbox in bboxes)  # y_min + height

    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


def mask_to_rle(mask):
    assert np.all(np.isin(mask, [0, 1])), "Binary mask contains values other than 0 and 1"
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle, height, width):
    '''
    rle: run-length as string formated (start length)
    height, width: dimensions of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T  # Needed to align to RLE direction


def polygon_to_binary_mask(points, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], color=1)
    return mask


def mask_to_annotation_representations(mask):
    height, width = mask.shape
    area = get_area(mask)

    bboxes = get_bboxes(mask)
    bbox = merge_bboxes(bboxes)

    rle = mask_to_rle(mask)
    rle = {
        'size': (height, width),
        'counts': rle
    }
    return area, bbox, rle
