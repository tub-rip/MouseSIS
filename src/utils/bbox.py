import numpy as np
import cv2
from shapely.geometry import Polygon

def crop_events_bbox(events: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Crop events. x in width, y in height direction

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        y0 (int): Start of the crop, at row[0]
        y1 (int): End of the crop, at row[0]
        x0 (int): Start of the crop, at row[1]
        x1 (int): End of the crop, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    mask = (
        (y0 <= events[..., 0])
        * (events[..., 0] < y1)
        * (x0 <= events[..., 1])
        * (events[..., 1] < x1)
    )
    cropped = events[mask]
    return cropped


def visualize_bboxes(bboxes, image, color=(0, 255, 0), thickness=1,
                     font_scale=0.5, no_decimals=True):
    """
    Visualize bounding boxes given as nx5 array on an image.
    Each row is a bbox in the format (x1, y1, x2, y2, score).
    Returns the image with rendered bounding boxes.
    """
    if isinstance(color, tuple):
        color = [color for bbox in bboxes]

    for bbox, c in zip(bboxes, color):
        x1, y1, x2, y2, score = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      c, thickness)
        score  = str(int(score)) if no_decimals else f'{score:.2f}'
        cv2.putText(image, f'{score}', (int(x1), int(y1) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, thickness)
        
    return image


def visualize_masks(masks, image, colors=None, alpha=0.5):
    """
    Visualize segmentation masks given as a list of binary masks on an image.
    Masks should be a list of 2D binary numpy arrays.
    Returns the image with rendered masks.
    """
    if colors is None:
        # Generate random colors if colors are not provided
        colors = np.random.randint(0, 256, size=(len(masks), 3), dtype=np.uint8)
    elif len(colors) != len(masks):
        raise ValueError("Number of masks should match the number of colors.")

    mask_image = np.zeros_like(image)

    for mask, color in zip(masks, colors):
        mask = mask.astype(bool)
        mask_image[mask] = color

    blended = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    return blended


def get_bbox_from_mask(mask):
    """
    Extracts the bounding box coordinates (x1, y1, x2, y2) from a binary mask.
    
    Parameters:
    mask (numpy.ndarray): Binary mask array where 1 represents the object and 0 represents the background.
    
    Returns:
    numpy.ndarray: Bounding box coordinates in the format (x1, y1, x2, y2).
    """
    non_zero_indices = np.nonzero(mask)
    y_coordinates, x_coordinates = non_zero_indices
    x1 = np.min(x_coordinates)
    y1 = np.min(y_coordinates)
    x2 = np.max(x_coordinates)
    y2 = np.max(y_coordinates)
    return np.array([x1, y1, x2, y2])


def polygon_to_boundingbox(points):
    # output bounding box has format [x_min, y_min, width, height]
    polygon = Polygon(points)
    x_min, y_min, x_max, y_max = polygon.bounds; 
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox