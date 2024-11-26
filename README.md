# Sketch Query Guided Object Detection

## Overview

This repository introduces a novel approach to **Sketch-guided Object Detection (SGOD)**, where multiple objects can be localized based on sketches with spatial awareness, improving the performance of object detection tasks in complex scenes. Traditional methods focus on detecting a single object at a time based on a single sketch, but this approach extends the functionality to detect multiple objects with meaningful spatial relationships between them. The method is built on top of the **DEtection TRansformers (DETR)** model, with significant modifications to integrate both photo and sketch features into the detection pipeline. The result is a more flexible, scalable, and spatially aware object detection system.

The core contribution of this research is to support **sketch query guidance**. This allows users to draw multiple sketches, which are then processed by the model to localize various objects with spatial alignment.

## Key Features

- **Multiple Object Detection with Spatial Awareness**: Users can query complex scenes and detect multiple objects while considering their spatial relationships (e.g., "dog to the right of a person").
- **Sketch-Guided Object Localization**: Unlike traditional object detection models, this approach uses sketches to guide the localization of objects in natural images.
- **DETR-based Architecture**: We leverage the DETR model with custom modifications to handle sketches in combination with photo features for object localization.

## Changes Made to the Original DETR Code

### 1. **Modifications in `models/detr.py`**

#### Original Code:
- **Feature Combination**: In the original DETR implementation, features from the backbone are processed and passed to the transformer only for single input images. In this work we modify DETR to process the input images along with sketch inputs.
  
#### New Additions:
- **Feature Combination**:
  ```python
  src = photo_features + sketch_features
  ```
  The primary modification here is the **element-wise addition** of `photo_features` and `sketch_features`. In the original DETR, the photo features would have been processed in isolation, but now both photo and sketch features are merged. This modification helps integrate the information from both the photo and the sketch inputs for object localization.We also compare other methods like concatenation with early/late fusion as well. 

- **Updated Forward Pass**:
  ```python
  hs = self.transformer(self.input_proj(photo), self.input_proj_(sketch), mask, self.query_embed.weight, pos[-1])[0]
  hs = self.transformer(self.input_proj(src), self.input_proj_(sketch), mask, self.query_embed.weight, pos[-1])[0]
  ```
  The forward pass has been updated to accept both the `photo` and `sketch` features. The transformer now processes both of these feature sets simultaneously, ensuring the model can understand and use both photo and sketch data for detection.

### 2. **Modifications in `models/transformer.py`**

#### Original Code:
- **Projection Layers**: The original code uses projection layers to map the input features to a suitable dimension for the transformer. It does not specifically account for handling both photo and sketch features in the same pipeline.

#### New Additions:
- **New Linear Layers**:
  ```python
  self.input_proj_ = nn.Linear(1000, 100)
  self.input_proj_ = nn.Linear(2100, 512)
  self.input_proj = nn.Linear(512, 100)
  ```
  These new layers are added to handle the feature projections of both the photo and sketch inputs. The layers are designed to handle the feature space after both inputs are combined.

- **Updated Target Tensor Size**:
  ```python
  target = torch.zeros(2000, bs, c)
  ```
  The target tensor size has been updated to 2000, as the model now processes a larger set of combined features from both the photo and sketch. This ensures that the model can handle more complex input data.

- **Modified Forward Pass to Handle Sketch**:
  The forward pass was adjusted to include both the photo and sketch features, with the encoder now accepting both feature types:
  ```python
  memory = self.encoder(src, sketch, src_key_padding_mask=mask, pos=pos_embed)
  ```
---

## Setup and Installation

To set up and run the project, follow these steps:

### Clone the Repository:

```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Preparation:

1. **Setup the Dataset:**
   Place the **sketchyCOCO dataset** in the directory `../sketch_detr_data/data/sketchyCOCO/Scene/`.

2. **Run the Data Preparation Script:**
   Run the script to prepare the dataset:
   ```bash
   python prepare_single_instance_sketch_data_COCO.py
   ```

   This will process the dataset and generate the necessary JSON files (`trainInTrain.json`, `valInTrain.json`) in the `sketch_retrieval_dataset/single_instance_dataset` folder.

3. **Verify Output:**
   After running the script, verify that the dataset is saved as a COCO JSON file and images are placed correctly in the output directory.

The script processes the sketch and bounding box annotations and prepares the data for training and evaluation.

Ensure you have the Sketch-guided Object Detection (SGOD) dataset in the following structure:
```
data/
  └── sketches_single_instance/
        ├── trainInTrain.json
        └── valInTrain.json

```
### Running the Code:

- **Inference**:
    ```bash
    python3 test.py --data_path data/Sketch/paper_version/val --resume checkpoint/checkpoint.pth
    ```

- **Evaluation**:
    ```bash
    python3 main.py --num_workers 14 --batch_size 4 --device "cuda:0" --eval --resume checkpoint/checkpoint.pth
    ```

- **Training**:
    ```bash
    python3 main.py --num_workers 14 --batch_size 4 --device "cuda:0" --resume detr-r50-e632da11.pth
    ```

---


## Results

The following images demonstrate the effectiveness of the **Sketch Query Guided Object Detection** approach. These visual results show how the model is able to detect objects and their spatial relationships based on sketches, as well as the architecture of the modified DETR model used in this study.

### Example 1: Multi-object Detection with Spatial Alignment

![Example 1](https://deepwilson.github.io/thesis_sketch_guided_detr/static/images/image_1.jpg)

This image shows how a sketch can guide the model to detect and localize objects in an image based on the user's query.

### Example 2: Detecting only sketch realted object despite having Multiple instance of same label

![Example 2](https://deepwilson.github.io/thesis_sketch_guided_detr/static/images/image_4.png)

In this example, the model successfully identifies only those instances drawn by user even though multiple objectsof same label existin the image, while preserving their spatial relationships as defined by the user’s sketch.

### Example 3: Detecting only sketch realted object despite having Multiple instance of same label

![Example 3](https://deepwilson.github.io/thesis_sketch_guided_detr/static/images/image_5.png)

Another example
### Example 4: Example of Spatial Alignment

![Example 4](https://deepwilson.github.io/thesis_sketch_guided_detr/static/images/image_6.png)

The model is capable of detecting multiple objects and maintaining the spatial arrangement as defined by the sketch input.

### Model Architecture

![Model Architecture](https://deepwilson.github.io/thesis_sketch_guided_detr/static/images/arch.png)

This image illustrates the architecture of the modified **DETR** model that incorporates sketch-guided object detection with spatial awareness. The key modification is the introduction of a **query canvas** for users to draw multiple instances, which enables the model to detect objects based on spatial alignment.

---


## Citation

If you find this work useful, please cite the following:

```
@thesis{Aricatt_Song_Chowdhury_2023, title={Sketch Query Guided Object Detection}, author={Aricatt, Deep  Wilson and Song, Yi-Zhe and Chowdhury, Pinaki  Nath}, year={2023}} 

```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

