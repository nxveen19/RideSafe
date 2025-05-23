import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import sys
sys.path.append('/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg/modules/ms_deform_attn.py')
sys.path.append('/home2/keshav06/TrafficViolations/deep-text-recognition-benchmark/modules')
sys.path.append('/home2/deepti.rawat/space_issue/home/keshav/deep-text-recognition-benchmark/')
# sys.path.insert(0, '/home2/keshav06/TrafficViolations/deep-text-recognition-benchmark')
from .lp_utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import os
import torchvision.transforms as transforms
from PIL import Image
import cv2
# import lp_utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from ultralytics import YOLO
import numpy as np
import cv2
from paddleocr import PaddleOCR

def deskew_plate(image):
    minangle = 0
    maxangle = 0
    min_area = 30*30
    average_h = 50
    image2 = image.copy()
    # print(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # use thresholding so that grey pixels are turned white and black pixels are turned black
    _, grayscale = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)


    # Find the contours in the image
    contours, hierarchy = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # print the contour
    # cv2.drawContours(image2, [max_contour], 0, (0, 255, 0), 3)
    # find the angle from the horizontal line of the contour and rotate the image
    try:
        rect = cv2.minAreaRect(max_contour)
    except:
        print(f"Skipping as no contour found")
        return None
    # crop the image to the bounding box of the contour from up and down
    top_most_point = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    bottom_most_point = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
    left_most_point = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    right_most_point = tuple(max_contour[max_contour[:, :, 0].argmax()][0])

    cropped_image = image[top_most_point[1]:bottom_most_point[1], left_most_point[0]:right_most_point[0]]
    image = cropped_image

    # print(rect)
    angle = rect[2]
    minangle = min(minangle, angle)
    maxangle = max(maxangle, angle)
    # angle is between 0 and 90, with 45 being the horizontal line
    if angle < 45: # if angle is less than 45, then it is the angle from the horizontal line to the right
        angle = angle
    else: # if angle is greater than 45, then it is the angle from the horizontal line to the left
        angle = angle - 90
    # print(angle)
    if image.shape[0] * image.shape[1] < min_area:
        print(f"Skipping as area is too small")
        return None
    image_height = image.shape[0]
    image_width = image.shape[1]
    if image_height > image_width or image_width/image_height > 2.5:
        print(f"Skipping as aspect ratio is too high")
        return None
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(rect[0], angle, 1), (image.shape[1], image.shape[0]))
    # cv2.imwrite(os.path.join(output_folder, file), rotated)
    # save grayscale image
    # cv2.imwrite(os.path.join(output_folder, file.replace('.png', '_contour.png')), image2)

    # increase the height of the upper half to have 3/5ths of the top half
    upper = rotated[:rotated.shape[0]* 3//5, :]
    lower = rotated[rotated.shape[0] * 2//5:, :]
    # ensure that both have same number of rows
    if upper.shape[0] != lower.shape[0]:
        lower = cv2.resize(lower, (upper.shape[1], upper.shape[0]))
    concatenated = cv2.hconcat([upper, lower])

    # image aspect ratio
    h = concatenated.shape[0]
    w = concatenated.shape[1]
    aspect_ratio = w/h
    if h < average_h:
    # resize the concatenated to have a height of average_h, bicubic interpolation
        concatenated = cv2.resize(concatenated, (int(average_h*aspect_ratio), average_h), interpolation=cv2.INTER_CUBIC)
    return concatenated

class LicensePlateModel:
    def __init__(self, model_path='best.pt'):
        """
        Initialize the License Plate model using YOLO for detection and PaddleOCR for recognition
        
        Args:
            model_path: Path to the YOLO model trained on 'bike', 'helmet', 'no-helmet', 'number-plate'
        """
        # Initialize YOLO model for detection
        self.detector = YOLO(model_path)
        
        # Initialize PaddleOCR for recognition
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Class index for number-plate in the YOLO model
        self.plate_class_idx = 3  # 'number-plate' is 4th class (index 3)
    
    def __call__(self, image):
        """
        Process an image to detect license plates and recognize text
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            lp_boxes: List of license plate bounding boxes [x1, y1, x2, y2]
            lp_texts: List of recognized texts
        """
        # Detect objects using YOLO
        results = self.detector(image)
        
        # Extract license plate detections
        lp_boxes = []
        lp_crops = []
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls[0].item())
                
                # Check if the detection is a license plate
                if cls == self.plate_class_idx:
                    xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Ensure coordinates are within image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    # Add some padding to the crop if possible
                    pad = 5
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
                    
                    # Extract license plate crop
                    lp_crop = image[y1:y2, x1:x2]
                    
                    if lp_crop.size > 0:  # Ensure the crop is valid
                        lp_boxes.append(xyxy)
                        lp_crops.append(lp_crop)
        
        # Recognize text in license plates
        lp_texts = []
        for crop in lp_crops:
            # PaddleOCR expects RGB image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            ocr_result = self.ocr.ocr(crop_rgb, cls=True)
            
            # Extract text from OCR result
            if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                text = ""
                for line in ocr_result[0]:
                    # Each line contains recognition result and confidence: [(text, confidence)]
                    if len(line) >= 2 and len(line[1]) >= 1:
                        text += line[1][0] + " "
                lp_texts.append(text.strip())
            else:
                lp_texts.append("")
        
        return np.array(lp_boxes), lp_texts
