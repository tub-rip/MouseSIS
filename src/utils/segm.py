import numpy as np
import cv2 as cv
from scipy.sparse import coo_matrix

def points_to_rle(image_shape, points):
    image = np.zeros(image_shape, dtype=np.uint8)
    points_int = [[int(x), int(y)] for x, y in points]
    cv.fillPoly(image, [np.array(points_int)], 255)
    rle = []
    last_color = 0
    for col in range(image.shape[1]):
        for row in range(image.shape[0]):
            if image[row, col] != last_color:
                if image[row, col] == 255: 
                    start_pixel_count = row + col * image.shape[0] 
                    rle.append(start_pixel_count)
                else: 
                    end_pixel_count = row + col * image.shape[0] 
                    rle.append(end_pixel_count-start_pixel_count)
                last_color = image[row, col]
    rle_string = ' '.join(map(str, rle))
    return rle_string

# def points_to_rle(image_shape, points):
#     image = np.zeros(image_shape, dtype=np.uint8)
#     points_int = np.round(points).astype(np.int32)
#     cv.fillPoly(image, [points_int], 255)
    
#     # Convert image to boolean mask
#     mask = image.astype(bool)
    
#     # Convert boolean mask to sparse COO matrix
#     sparse_matrix = coo_matrix(mask)
    
#     # Find the start and length of each run of ones in the COO matrix
#     rle = []
#     current_col = 0
#     for row, col in zip(sparse_matrix.row, sparse_matrix.col):
#         if col != current_col:
#             rle.append(row)
#         current_col = col
#     rle.append(len(sparse_matrix.data) - rle[-1])
    
#     rle_string = ' '.join(map(str, rle))
#     return rle_string



def polygon_to_area(polygon):
    area = 0
    x = [point[0] for point in polygon]
    y = [point[1] for point in polygon]
    area += 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def rle_to_mask(image_shape, rle_string):
    image = np.zeros(image_shape, dtype=np.uint8)
    if isinstance(rle_string, list): 
        regions = rle_string
    elif isinstance(rle_string, str): 
        regions = rle_string.split()
    
    for i in range(0, len(regions), 2):
        start_pixel = int(regions[i]) 
        length = int(regions[i + 1])
        start_col,start_row = divmod(start_pixel - 1, image_shape[0])  
        end_col,end_row  = divmod(start_pixel + length - 1, image_shape[0])  
        image[start_row:end_row + 1, start_col:end_col + 1] = 255
    return image


def merge_masks(rle1, rle2,img_shape):
    mask1 = rle_to_mask(img_shape, rle1)
    mask2 = rle_to_mask(img_shape, rle2)
    union = np.logical_or(mask1, mask2)
    intersection = np.logical_and(mask1, mask2)
    merged_mask = np.where(intersection, 1, union).astype(np.uint8)


    rle = []
    last_color = 0
    for col in range(merged_mask.shape[1]):
        for row in range(merged_mask.shape[0]):
            if merged_mask[row, col] != last_color:
                if merged_mask[row, col] == 1: 
                    start_pixel_count = row + col * merged_mask.shape[0] 
                    rle.append(start_pixel_count)
                else: 
                    end_pixel_count = row + col * merged_mask.shape[0] 
                    rle.append(end_pixel_count-start_pixel_count)
                last_color = merged_mask[row, col]
    rle_string = ' '.join(map(str, rle))
    return rle_string