import ultralytics
import numpy as np
from transformers import SamModel, SamProcessor
import torch
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.savefig("boxes_on_image.png")


def show_masks_on_image(raw_image, masks, scores, name='masks_on_image'):
    num_instances, nb_predictions, height, width = masks.shape

    # Create subplots
    if num_instances == 1 and nb_predictions == 1:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        axes = np.array([[ax]])
    elif num_instances == 1:
        fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 5))
        axes = np.expand_dims(axes, 0)
    elif nb_predictions == 1:
        fig, axes = plt.subplots(num_instances, 1, figsize=(15, 5 * num_instances))
        axes = np.expand_dims(axes, 1)
    else:
        fig, axes = plt.subplots(num_instances, nb_predictions, figsize=(15, 5 * num_instances))

    for instance_idx in range(num_instances):
        for pred_idx in range(nb_predictions):
            mask = masks[instance_idx, pred_idx].cpu().detach()
            score = scores[instance_idx, pred_idx].item()
            ax = axes[instance_idx, pred_idx]
            ax.imshow(np.array(raw_image))
            show_mask(mask, ax)
            ax.title.set_text(f"Instance {instance_idx+1}, Mask {pred_idx+1}, Score: {score:.3f}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{name}.png")


class SamYoloDetector:
    def __init__(self, yolo_path, device='cuda:0') -> None:
        # self.detector = ultralytics.YOLO(yolo_path, verbose=False)
        self.detector = ultralytics.YOLO(yolo_path)
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.device = device

    def run(self, img):
        result = self.detector(img)[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy() # x1, y1, x2, y2
        scores = result.boxes.conf.detach().cpu().numpy()

        if not len(boxes):
            return None, None

        #dets = np.zeros((len(boxes), 5), dtype=np.float32)
        #dets[:,:4] = boxes
        #dets[:, 4] = scores

        boxes_list = [[boxes.tolist()]]
        inputs = self.sam_processor(img.transpose(2, 0, 1), input_boxes=[boxes_list], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        iou_scores = outputs.iou_scores.cpu()[0]

        num_instances, nb_predictions, height, width = masks.shape
        max_indices = iou_scores.argmax(dim=1, keepdim=True)
        gather_indices = max_indices[..., None, None].expand(-1, 1, height, width)
        selected_masks = torch.gather(masks, 1, gather_indices).squeeze(1)

        #show_boxes_on_image(img, boxes)
        #show_masks_on_image(img, masks, iou_scores)
        #show_masks_on_image(img, selected_masks[:, None], iou_scores, name='selected_masks_on_image')
        return selected_masks.cpu().numpy(), scores