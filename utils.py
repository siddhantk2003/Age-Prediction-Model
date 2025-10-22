"""
Utility functions:
- UTKFaceDataset: loads images from UTKFace folder (filename format: age_gender_race_date.jpg)
- transforms and preprocessing helpers
- download of OpenCV DNN face detector files if missing
"""
import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import urllib.request

def parse_age_from_filename(filename):
    # UTKFace filenames: [age]_[gender]_[race]_[date].jpg
    base = os.path.basename(filename)
    m = re.match(r'^(\d+)_', base)
    if not m:
        raise ValueError(f"Could not parse age from filename: {filename}")
    return int(m.group(1))

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        """
        root_dir: directory containing UTKFace images
        transform: torchvision transforms to apply to each image
        limit: optional integer to limit dataset size (useful for quick experiments)
        """
        self.root_dir = root_dir
        self.transform = transform
        files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()
        if limit:
            files = files[:limit]
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        age = parse_age_from_filename(path)
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # regression target as float tensor
        return image, torch.tensor([float(age)], dtype=torch.float32)

def default_transforms(train=False):
    """
    Returns transforms suitable for EfficientNet input (224x224).
    For CPU training we keep them simple.
    """
    size = 224
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

# OpenCV DNN face detector download helper
def ensure_face_detector(detector_dir):
    os.makedirs(detector_dir, exist_ok=True)

    proto_path = os.path.join(detector_dir, "deploy.prototxt")
    model_path = os.path.join(detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    # ✅ Correct, active URLs
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    # download .prototxt if missing
    if not os.path.exists(proto_path):
        print(f"Downloading prototxt to {proto_path}...")
        urllib.request.urlretrieve(proto_url, proto_path)
        print("Downloaded: deploy.prototxt")

    # download .caffemodel if missing
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Downloaded: res10_300x300_ssd_iter_140000.caffemodel")

    return proto_path, model_path

