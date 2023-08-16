import os
import cv2
import glob
import shutil
from scipy.io import loadmat
from scipy.stats import mode
import numpy as np
import os
import cv2
import numpy as np
from scipy.io import loadmat
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.patches as patches
import json
from tqdm.notebook import tqdm


# Define paths and directories
data_dir = '../sketch_detr_data/data/sketchyCOCO/Scene/'

split = 'trainInTrain'  # or 'trainInTrain'

instance_path = 'Annotation/paper_version/{}/INSTANCE_GT'.format(split)
bbox_path = 'Annotation/paper_version/{}/BBOX'.format(split)
class_path = 'Annotation/paper_version/{}/CLASS_GT'.format(split)

sketch_dir = 'Sketch/paper_version/{}'.format(split)
photo_dir = 'GT/{}'.format(split)
output_dir = 'sketch_retrieval_dataset/single_instance_dataset' 
output_sketch_dir = f'single_instance_sketches_{split}'

coco_labels = {0: 'unlabelled', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


# Load mapping
sketchycoco_2_cocostuff = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 10, 6: 11, 7: 17, 8: 18, 9: 19, 10: 20,
                           11: 21, 12: 22, 13: 0, 14: 24, 15: 25, 16: 106, 17: 124, 18: 106, 19: 169}

os.makedirs(os.path.join(output_dir, output_sketch_dir), exist_ok=True)


# Create an empty list to store COCO annotations
annotations = []
image_infos = []
all_image_ids = os.listdir(os.path.join(data_dir, instance_path))
for image_id in tqdm(all_image_ids, total=len(all_image_ids)):
    # image_id = "000000004229.mat"
    # image_id = "000000061867.mat"
    # image_id = "000000241347.mat"
    # print(image_id)

    
    
    instance_gt = loadmat(os.path.join(data_dir, instance_path, image_id))['INSTANCE_GT']
    bbox_gt = loadmat(os.path.join(data_dir, bbox_path, image_id))['BBOX']
    class_gt = loadmat(os.path.join(data_dir, class_path, image_id))['CLASS_GT']

    sketch_gt_path = os.path.join(data_dir, sketch_dir, image_id.replace('.mat', '.png'))
    sketch_gt = cv2.imread(sketch_gt_path)
    photo_gt_path = os.path.join(data_dir, photo_dir, image_id.replace('.mat', '.png'))
    photo_gt = cv2.imread(photo_gt_path)
    
    unique_instance_gt = np.unique(instance_gt)

    flag = 0
    bbox_counter = 0
    for instance in unique_instance_gt:     
        class_id = class_gt[np.where(instance_gt == instance)]
        assert class_id[0] == class_id.mean() # should have only one value

        class_id = class_id[0]
        if class_id == 0 or class_id > 15: #0: background, >15: background(sky, grass etc.), 1-15: foreground(zebra. elephant etc.)
            continue
        else:
            bbox_counter+=1
        xids, yids = np.where(instance_gt == instance) # sketch bbox
        xmin, xmax, ymin, ymax = xids.min(), xids.max(), yids.min(), yids.max()

        xids, yids = np.where(bbox_gt[:,:,bbox_counter] == class_id) # coco bbox
        coco_xmin, coco_xmax, coco_ymin, coco_ymax = xids.min(), xids.max(), yids.min(), yids.max()

        
        if xmax - xmin <1 or ymax - ymin < 1:
            continue
        
        class_name = coco_labels[sketchycoco_2_cocostuff[class_id]]
        class_name = class_name.replace(' ', '-')
        sketch_patch = sketch_gt[xmin:xmax, ymin:ymax]
        
        
        
        # cv2.imwrite(os.path.join(output_sketch_dir, '{}_{}_small.png'.format(instance, class_name)), sketch_patch)

        canvas = np.ones_like(sketch_gt)*255
        canvas[xmin:xmax, ymin:ymax] = sketch_patch

        # cv2.imwrite(os.path.join(output_sketch_dir, '{}_{}_large.png'.format(instance, class_name)), canvas)

        # Create a rectangle patch
        bbox_rect = patches.Rectangle(
            (coco_ymin, coco_xmin),  # (x, y) coordinates of the lower left corner
            coco_ymax - coco_ymin,  # Width
            coco_xmax - coco_xmin,  # Height
            linewidth=1,
            edgecolor='r',
            facecolor='none'  # Transparent rectangle fill
        )



        final_image = canvas
        filename = image_id.replace(".mat", f"_{bbox_counter}.png")
        cv2.imwrite(os.path.join(output_dir, output_sketch_dir, filename), canvas)
        id = int(image_id.replace(".mat", f"{bbox_counter}"))
        final_bbox = [coco_xmin, coco_ymin, coco_xmax, coco_ymax]
        x = int(coco_xmin)
        y = int(coco_ymin)
        width =  int(coco_xmax - coco_xmin)
        height = int(coco_ymax - coco_ymin)

        final_bbox = [x, y, width, height]
        coco_class_id = sketchycoco_2_cocostuff[class_id]
        coco_area = int(width * height)
        
        # Create a dictionary to store image information
        image_info = {
            "id": id,
            "file_name": filename,
            "width": final_image.shape[1],
            "height": final_image.shape[0],
        }
        image_infos.append(image_info)

        # Create a list to store annotations for this image
        image_annotations = []
        
        # Create a dictionary to store annotation for this bbox
        ann_len = len(annotations) + 1
        annotation = {
            "id": ann_len,
            "image_id": id,
            "category_id": coco_class_id,
            "bbox": final_bbox,
            "segmentation": [],  
            "area": coco_area,
            "iscrowd": 0,  # Set to 0 for regular objects
        }
        image_annotations.append(annotation)
        annotations.append(annotation)
        
        # # Display the two plots side by side
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # # Plot the original photo with bbox
        # axs[1].imshow(photo_gt)
        # axs[1].axis("off")
        # axs[1].add_patch(bbox_rect)  # Add bbox rectangle to the photo plot
        # axs[1].set_title("Original Photo with Bounding Box")

        # # Plot the canvas with sketch
        # axs[0].imshow(canvas)
        # axs[0].axis("off")
        # axs[0].set_title("Canvas with Sketch Patch")

        # # Adjust layout and display the plots
        # plt.tight_layout()
        # plt.show()
            
    # break




# Create the COCO dataset dictionary
coco_dataset = {
    "info": "",
    "images": [i for i in image_infos],
    "categories": [{"id":key, "name":value, "supercategory": "object"} for key,value in coco_labels.items()],
    "annotations": annotations
}

print(coco_dataset)

# Save the COCO dataset as a JSON file
coco_save_path = os.path.join(output_dir, f"single_instance_{split}.json")
with open(coco_save_path, "w") as f:
    json.dump(coco_dataset, f)

print("COCO dataset JSON file created successfully.")
