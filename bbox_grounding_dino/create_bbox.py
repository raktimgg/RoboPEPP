import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
import tqdm
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict



"""
Hyper parameters
"""
TEXT_PROMPT = "robotic arm ."
IMG_PATH = "notebooks/images/truck.jpg"
IMG_PATH = "/data/raktim/Datasets/dream/synthetic/panda_synth_test_dr/001331.rgb.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real/panda-3cam_azure/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real_annotated/panda-3cam_azure/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real/panda-3cam_kinect360/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real_annotated/panda-3cam_kinect360/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real/panda-3cam_realsense/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real_annotated/panda-3cam_realsense/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real/panda-orb/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real_annotated/panda-orb/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real/franka_right_crrl/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real_annotated/franka_right_crrl/"

IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real/franka_left_crrl/"
SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/real_annotated/franka_left_crrl/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic/panda_synth_test_dr/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic_annotated/panda_synth_test_dr/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic/panda_synth_test_photo/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic_annotated/panda_synth_test_photo/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic/kuka_synth_test_dr/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic_annotated/kuka_synth_test_dr/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic/kuka_synth_test_photo/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic_annotated/kuka_synth_test_photo/"

# IMAGE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic/baxter_synth_test_dr/"
# SAVE_FOLDER = "/data/raktim/Projects/JEPA/Holistic-Robot-Pose-Estimation/data/dream/synthetic_annotated/baxter_synth_test_dr/"

file_list = sorted(os.listdir(IMAGE_FOLDER))
file_list = [file for file in file_list if file.endswith(".jpg")]

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


for idx, file in tqdm.tqdm(enumerate(file_list), total=len(file_list)):
    # if idx < 550:
    #     continue
    # print(idx, len(file_list))
    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT
    img_path = os.path.join(IMAGE_FOLDER, file)

    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    if "panda_synth_test_dr" in IMAGE_FOLDER or "kuka_synth_test_dr" in IMAGE_FOLDER:
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        max_ind = np.argmax(confidences)
        wmin, hmin, wmax, hmax = xyxy[max_ind][0], xyxy[max_ind][1], xyxy[max_ind][2], xyxy[max_ind][3]
        if wmax-wmin < 130 and hmax-hmin < 130:
            boxes, confidences, labels = predict(
                                    model=grounding_model,
                                    image=image,
                                    caption='large robot .',
                                    box_threshold=BOX_THRESHOLD,
                                    text_threshold=TEXT_THRESHOLD,
                                )
            if confidences.shape[0] == 0:
                boxes, confidences, labels = predict(
                                    model=grounding_model,
                                    image=image,
                                    caption='large robot .',
                                    box_threshold=BOX_THRESHOLD-0.1,
                                    text_threshold=TEXT_THRESHOLD-0.1,)
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            max_ind = np.argmax(confidences)
            wmin, hmin, wmax, hmax = xyxy[max_ind][0], xyxy[max_ind][1], xyxy[max_ind][2], xyxy[max_ind][3]
            if wmax-wmin < 130 and hmax-hmin < 130:
                wmin, wmax = 150, 640-150
                hmin, hmax = 100, 480-100
        input_boxes = np.array([[wmin, hmin, wmax, hmax]])
        # print(confidences, confidences[max_ind])
        confidences = torch.Tensor([confidences[max_ind]])
        labels = [labels[max_ind]]
    else:
        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # if torch.cuda.get_device_properties(0).major >= 8:
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # masks, scores, logits = sam2_predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_boxes,
    #     multimask_output=False,
    # )

    # """
    # Post-process the output of the model to get the masks, scores, and logits for visualization
    # """
    # # convert the shape to (n, H, W)
    # if masks.ndim == 4:
    #     masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    max_ind = np.argmax(confidences)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    # save the input_boxes as json
    json_results = {
        "bonuding_boxes": input_boxes[max_ind].tolist(),
    }
    save_name = os.path.join(SAVE_FOLDER, f"{file}").replace(".jpg", ".json")
    with open(save_name, "w") as f:
        json.dump(json_results, f)

    # # save the masks as jpg images
    # mask = masks[max_ind].astype(np.uint8) * 255
    # # mask = cv2.resize(mask, (w, h))
    # cv2.imwrite(os.path.join(SAVE_FOLDER, file), mask)