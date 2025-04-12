# Setup Instructions for Bounding Box Annotation with GroundingDINO

During testing, we leverage the GroundingDINO framework from the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repository to predict bounding boxes. While our test pipeline supports real-time bounding box prediction, we recommend pre-processing the test dataset using the steps outlined below. This significantly reduces overhead during evaluation and enables faster, more efficient testing.

1. Clone the Grounded-SAM-2 Repository
```
cd bbox_grounding_dino
git clone https://github.com/IDEA-Research/Grounded-SAM-2
```

2. Install Dependencies and Set Up Models
```
export CUDA_HOME=/path/to/cuda-12.1/
cd Grounded-SAM-2/
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install supervision pycocotools transformers addict yapf timm
```
**Download pretrained mode**: Download the trained model `groundingdino_swint_ogc.pth` from the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repo and store it inside `gdino_checkpoints` directory.
```
Grounded-SAM-2/
├── grounding_dino/
├── gdino_checkpoints/
│   └── groundingdino_swint_ogc.pth
```

3. Prepare and Run Bounding Box Script
Copy the `create_bbox.py` script:
```
cp ../create_bbox.py .
```
**Modify the script**:
Edit line 33 in `create_bbox.py` to set the correct image folder path:
```
IMAGE_FOLDER = "/path/to/your/image/folder"
```
**Run the script**:
```
python create_bbox.py
```
This will generate annotated `.json` files containing bounding box predictions, ready for use with RoboPEPP's test pipeline.