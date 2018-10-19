import pandas as pd 
import numpy as np
import scipy.misc
import pydicom 
import glob
import sys
import os

from scipy.ndimage.interpolation import zoom

!git clone https://github.com/fizyr/keras-retinanet
os.chdir("keras-retinanet") 
!python setup.py build_ext --inplace

DATA_DIR = "/kaggle/input/"
ROOT_DIR = "/kaggle/working/"
# I converted training set DICOMs to PNGs, it should be part of the data environment
train_pngs_dir = os.path.join(DATA_DIR, "rsna-pneu-train-png/stage_1_train_pngs/orig/")
test_dicoms_dir  = os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_1_test_images/") 

# Create annotations for RetinaNet training
import pandas as pd 

bbox_info = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_1_train_labels.csv"))
detailed_class_info = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv"))
detailed_class_info = detailed_class_info.drop_duplicates()

# To get started, we'll train on positives only
positives = detailed_class_info[detailed_class_info["class"] == "Lung Opacity"]
# Annotations file should have no header and columns in the following order:
# filename, x1, y1, x2, y2, class 
positives = positives.merge(bbox_info, on="patientId")
positives = positives[["patientId", "x", "y", "width", "height", "Target"]]
positives["patientId"] = [os.path.join(train_pngs_dir, "{}.png".format(_)) for _ in positives.patientId]
positives["x1"] = positives["x"] 
positives["y1"] = positives["y"] 
positives["x2"] = positives["x"] + positives["width"]
positives["y2"] = positives["y"] + positives["height"]
positives["Target"] = "opacity"
del positives["x"], positives["y"], positives["width"], positives["height"]

# If you want to add negatives, follow the same format as above, except put NA for x, y, width, and height
annotations = positives

# Before we save to CSV, we have to do some manipulating to make sure
# bounding box coordinates are saved as integers and not floats 
# Note: This is only necessary if you include negatives in your annotations
annotations = annotations.fillna(88888)
annotations["x1"] = annotations.x1.astype("int32").astype("str") 
annotations["y1"] = annotations.y1.astype("int32").astype("str") 
annotations["x2"] = annotations.x2.astype("int32").astype("str") 
annotations["y2"] = annotations.y2.astype("int32").astype("str")
annotations = annotations.replace({"88888": None}) 
annotations = annotations[["patientId", "x1", "y1", "x2", "y2", "Target"]]
annotations.to_csv(os.path.join(ROOT_DIR, "annotations.csv"), index=False, header=False)

# We also need to save a file containing the classes
classes_file = pd.DataFrame({"class": ["opacity"], "label": [0]}) 
classes_file.to_csv(os.path.join(ROOT_DIR, "classes.csv"), index=False, header=False) 

# ImageNet pre-trained ResNet50 backbone 
# Image size: 256 x 256
# Batch size: 1
# Epochs: 1
# Steps per epoch: 1,000
# Data augmentation
!python /kaggle/working/keras-retinanet/keras_retinanet/bin/train.py --backbone "resnet50" --image-min-side 256 --image-max-side 256 --batch-size 1 --random-transform --epochs 1 --steps 1000 csv /kaggle/working/annotations.csv /kaggle/working/classes.csv

# Convert model 
!python /kaggle/working/keras-retinanet/keras_retinanet/bin/convert_model.py /kaggle/working/keras-retinanet/snapshots/resnet50_csv_01.h5 /kaggle/working/keras-retinanet/converted_model.h5 

# Load converted model
from keras_retinanet.models import load_model 

retinanet = load_model(os.path.join(ROOT_DIR, "keras-retinanet/converted_model.h5"), 
                       backbone_name="resnet50")
                       
# Preprocessing function 
def preprocess_input(x):
    x = x.astype("float32")
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.680
    return x

test_dicoms = glob.glob(os.path.join(test_dicoms_dir, "*.dcm"))
test_patient_ids = [_.split("/")[-1].split(".")[0] for _ in test_dicoms]
test_predictions = [] 
for i, dcm_file in enumerate(test_dicoms): 
    sys.stdout.write("Predicting images: {}/{} ...\r".format(i+1, len(test_dicoms)))
    sys.stdout.flush() 
    # Load DICOM and extract pixel array 
    dcm = pydicom.read_file(dcm_file)
    arr = dcm.pixel_array
    # Make 3-channel image
    img = np.zeros((arr.shape[0], arr.shape[1], 3))
    for channel in range(img.shape[-1]):
        img[..., channel] = arr 
    # Resize 
    # Change image size if necessary!
    scale_factor = 256. / img.shape[0]
    img = zoom(img, [scale_factor, scale_factor, 1], order=1, prefilter=False)
    # Preprocess with ImageNet mean subtraction
    img = preprocess_input(img) 
    prediction = retinanet.predict_on_batch(np.expand_dims(img, axis=0))
    test_predictions.append(prediction)   
    
# Extract predictions
test_pred_df = pd.DataFrame() 
for i, pred in enumerate(test_predictions):
    # Take top 5 
    # Should already be sorted in descending order by score
    bboxes = pred[0][0][:5]
    scores = pred[1][0][:5]
    # -1 will be predicted if nothing is detected
    detected = scores > -1 
    if np.sum(detected) == 0: 
        continue
    else:
        bboxes = bboxes[detected]
        bboxes = [box / scale_factor for box in bboxes]
        scores = scores[detected]
    individual_pred_df = pd.DataFrame() 
    for j, each_box in enumerate(bboxes): 
        # RetinaNet output is [x1, y1, x2, y2] 
        tmp_df = pd.DataFrame({"patientId": [test_patient_ids[i]], 
                               "x": [each_box[0]],  
                               "y": [each_box[1]], 
                               "w": [each_box[2]-each_box[0]],
                               "h": [each_box[3]-each_box[1]],
                               "score": [scores[j]]})
        individual_pred_df = individual_pred_df.append(tmp_df) 
    test_pred_df = test_pred_df.append(individual_pred_df) 

test_pred_df.head()

# Generate submission

# Set box threshold for inclusion
threshold = 0.35

list_of_pids = [] 
list_of_preds = [] 
for pid in np.unique(test_pred_df.patientId): 
    tmp_df = test_pred_df[test_pred_df.patientId == pid]
    tmp_df = tmp_df[tmp_df.score >= threshold]
    # Skip if empty
    if len(tmp_df) == 0:
        continue
    predictionString = " ".join(["{} {} {} {} {}".format(row.score, row.x, row.y, row.w, row.h) for rownum, row in tmp_df.iterrows()])
    list_of_preds.append(predictionString)
    list_of_pids.append(pid) 

positives = pd.DataFrame({"patientId": list_of_pids, 
                          "PredictionString": list_of_preds}) 

negatives = pd.DataFrame({"patientId": list(set(test_patient_ids) - set(list_of_pids)), 
                          "PredictionString": [""] * (len(test_patient_ids)-len(list_of_pids))})

submission = positives.append(negatives) 
