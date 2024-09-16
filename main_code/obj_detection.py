import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import supervision as sv

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

import pdb

# viz functions--------------------------------------------------------------------------------------------------------------------
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------

class detection():
    def __init__(self, img, text_promt=None):
        self.img = img
        
        if text_promt is not None:
            self.text_promt = text_promt
        else:
            self.text_promt = ["soup_can", "banana", "bottle", "snack_box", "color_object"]


        self.yolo_model = None
        self.sam_model = None

        self.yolo_setup()
        self.sam_setup()

    def yolo_setup(self):
        self.yolo_model = YOLOWorld(model_id="yolo_world/l")

    def yolo_detect(self, visual=False):
        # classes = ["banana"]
        self.yolo_model.set_classes(self.text_promt)
        results = self.yolo_model.infer(self.img)
        detections = sv.Detections.from_inference(results)
        print("detected objects :", detections.data['class_name'])
        if visual:
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
            LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

            annotated_image = self.img.copy()
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
            sv.plot_image(annotated_image, (20, 20))
        
        return detections.xyxy

    def sam_setup(self):
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        checkpoint = "../segment-anything-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.sam_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    def sam_detect(self, bound_boxs, visual=False):
        if bound_boxs.shape[0] == 0:
            print("No bounding box given")
            return

        self.sam_model.set_image(self.img)

        input_point = input("obj_points :")
        pdb.set_trace()
        input_lable = [*range()]

        masks, scores, _ = self.sam_model.predict(
            point_coords=input_point,
            point_labels=None,
            box=bound_boxs,
            multimask_output=False,
        )

        if visual:
            image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) 
            plt.figure(figsize=(20, 20))
            plt.imshow(image)

            if masks.shape[0] == 1:
                show_mask(masks.squeeze(0), plt.gca(), random_color=True)
            else:
                for mask in masks:
                    show_mask(mask.squeeze(0), plt.gca(), random_color=True)
            for box in bound_boxs:
                show_box(box, plt.gca())
            plt.axis('off')
            plt.show()

        return masks

    def get_seg(self, yolo_viz=False, sam_viz=False, mask_viz=False):
        bbox = self.yolo_detect(yolo_viz)
        masks = self.sam_detect(bbox, sam_viz)

        if masks.shape[0] == 1:
            plt.figure(figsize=(20, 20))
            plt.imshow(masks.squeeze(0))
            plt.show()
        else:
            for mask in masks:
                plt.figure(figsize=(20, 20))
                plt.imshow(mask.squeeze(0))
                plt.show()

        return masks


if __name__ == "__main__":
    img_name = "test_img/test_rgb_1.jpg"
    img_name = "test_img/test_rgb_save1.jpg"
    img_name = "test_img/test_rgb_3456.jpg"
    # img_name = "test_img/test_rgb_10001.jpg"
    # img_name = "test_img/test_rgb_778.jpg"
    # img_name = "test_img/test_rgb_3455.jpg"
    # img_name = "test_img/master_chef.png"
    # img_name = "test_img/soup_can.jpg"

    img_name = "/home/j0k/Project/Imsa/main_code/test_data/test_real_experiment/images/test_rgb_1.jpg"
    img = cv2.imread(img_name)
    det = detection(img)
    det.get_seg(yolo_viz=True, sam_viz=True)

