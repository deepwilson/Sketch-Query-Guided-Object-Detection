# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.coco import make_coco_transforms as make_face_transforms

import matplotlib.pyplot as plt
import time


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='face')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='eval_multiple_sketch_instances',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.1, type=float) #0.02 for multiple instances

    return parser

from pycocotools.coco import COCO



def get_target_frim_image_id(image_id, sketch, coco):
    sketch_np = np.array(sketch)
    new_sketch = np.ones_like(sketch_np)* 255

    # image_id = 12345
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    target = coco.loadAnns(annotation_ids)
    print(target)
    # print(target.keys())
    # for annotation in target['annotations']:
    for annotation in target[:1]:
        bbox = annotation['bbox']
        x, y, w, h = [int(coord) for coord in bbox]
        print("&&&&&&&&&&&", x, y, w, h)
        new_sketch[y:y+h, x:x+w] = sketch_np[y:y+h, x:x+w]
        plt.imshow(new_sketch)
        plt.show()
    new_sketch = Image.fromarray(new_sketch)
    return new_sketch

@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    # coco_annotation_file = 'data/single_instance_valInTrain.json'
    # coco = COCO(coco_annotation_file)

    
    
    model.eval()
    duration = 0
    for img_sample in images_path[:50]:
        # img_sample = os.path.join(args.data_path, "000000572886.png")
        # img_sample = os.path.join("../sketch_detr/sketch_retrieval_dataset/sketches_single_instance/valInTrain", "000000557252_3.png")
        # img_sample = os.path.join("../sketch_detr/sketch_retrieval_dataset/testing_sketches_single_instance/valInTrain", "000000557252.png")
        # img_sample = os.path.join("../sketch_detr/sketch_retrieval_dataset/testing_sketches_single_instance/valInTrain", "000000557252.png")
        # img_sample = os.path.join("data/Sketch/paper_version/valInTrain", "000000557252.png")
        filename = os.path.basename(img_sample)
        if "_" in  filename:
            photo = filename.split("_")[0]+".png" #for single sketch instance
        else:
            photo = filename
        
        # photo = filename.split("_")[0]+".png" #for multiple instances (normal case)
        photo = os.path.join("data/GT/valInTrain/", photo)
        print(f"{photo:*^100}")
        print(f"{img_sample:*^100}")

        # filename = os.path.basename(img_sample)
        # img_sample = "data/GT/valInTrain/000000022718.png"
        # filename = "000000022718.png"
        # print("processing...{}".format(filename))
        # # open photo
        orig_image = Image.open(photo)
        w, h = orig_image.size
        transform = make_face_transforms("val")

        # open sketch
        # sketch_path = os.path.join("data/Sketch/paper_version/valInTrain", filename)
        sketch_path = img_sample
        sketch_ = Image.open(sketch_path)
        image_id = os.path.basename(sketch_path)
        image_id = (image_id.replace("_", ""))
        image_id = int(image_id.replace(".png", ""))
        print(f"image_id: {image_id}")
        # sketch_ = get_target_frim_image_id(image_id, sketch_, coco)
        w, h = orig_image.size

        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        try:
            image, targets = transform(orig_image, dummy_target)
        except Exception as e:
            print(e)
            continue
        image = image.unsqueeze(0)
        image = image.to(device)

        sketch, _ = transform(sketch_, dummy_target)
        sketch = sketch.unsqueeze(0)
        sketch = sketch.to(device)
        


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        start_t = time.perf_counter()
        outputs = model(image, sketch)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()
        # print(probas)

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        if len(bboxes_scaled) == 0:
            print("no boxes found :(")
            continue
        else:
            print(f"num of bboxes: {len(bboxes_scaled)}")

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sketch = np.array(sketch_)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

        img_save_path = os.path.join(output_path, filename)
        # cv2.imwrite(img_save_path, img)
        # Save the combined image
        # print(f"shapes img, sketch: {img.shape}, {sketch.shape} ")
        combined_image = np.concatenate((sketch, img), axis=1)
        img_save_path = img_save_path.replace(".png", "combined_image.jpg")
        print(img_save_path, cv2.imwrite(img_save_path.replace(".png", "combined_image.png"), combined_image))
        # import sys 
        # sys.exit()
        # cv2.imshow("img", img)
        # cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        print("Processed...{} ({:.3f}s)".format(img_save_path, infer_time))
        # break
    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(args.data_path)

    infer(image_paths, model, postprocessors, device, args.output_dir)

        