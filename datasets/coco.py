import sys
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (project root) and add it to the sys.path
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(sys.path)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from PIL import Image
import os

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, photo_path, sketch_path, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(sketch_path, ann_file) #(img_path, ann_file)
        self._transforms = transforms
        self.sketch_path = sketch_path
        self.photo_path = photo_path
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        # photo, sketch, target = super(CocoDetection, self).__getitem__(idx)
        photo, sketch = self._load_image(idx)
        target = self._load_target(idx)
        image_id = self.ids[idx] #from init COCOdetetion
        target = {'image_id': image_id, 'annotations': target}
        photo, sketch, target = self.prepare(photo, sketch, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return photo, sketch, target
    
    def _load_image(self, idx):
        path = self.coco.loadImgs(idx)[0]["file_name"]
        photo = Image.open(os.path.join(self.photo_path, path)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_path, path)).convert("RGB")
        return photo, sketch
    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, photo, sketch, target):
        w, h = sketch.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return photo, sketch, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "GT/trainInTrain", root / "Sketch/paper_version/trainInTrain", root / 'train.json'),
        "val": (root / "GT/valInTrain", root / "Sketch/paper_version/valInTrain", root / 'val.json'),
    }

    photo_path, sketch_path, ann_file = PATHS[image_set]
    print(f"photo_path, sketch_path {photo_path, sketch_path}")
    # dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    dataset = CocoDetection(photo_path, sketch_path, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset

if __name__ == "__main__":
    

    import argparse
    from main import get_args_parser
    parser = argparse.ArgumentParser('DETR dataset testing', parents=[get_args_parser()])
    args = parser.parse_args()

    # Step 2: Test the dataset for "train" and "val" image sets
    for image_set in ["train", "val"]:
        dataset = build(image_set, args)

        # Step 3: Print some shapes of tensors
        print(f"\nDataset - {image_set.capitalize()} set:")
        print(f"Number of samples: {len(dataset)}")
        print(f"Sample image and annotation information:")
        for i in range(min(5, len(dataset))):
            image, target = dataset[i]
            print(f"Sample {i + 1} - Image shape: {image.shape}, Annotations: {target}")


