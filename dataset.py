import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob

class AnomalyDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=518):
        """
        Parses a YOLO-style dataset format containing images and labels.
        In unsupervised anomaly detection, train should ideally contain only 'good' (normal) images.
        Here we define 'good' (0) as images having empty label files, and 'anomaly' (1) as having annotations.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.samples = []
        self.labels = [] # 0 for good, 1 for anomaly
        
        # DINOv2 uses standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # If it's a YOLO format dataset, we check train/images and test/images
        # Since this dataset has very few normal images, if is_train=True we will just load whatever normal images exist in train.
        # If is_train=False, we load everything from valid and test splits.
        splits = ['train'] if is_train else ['valid', 'test']
        
        for split in splits:
            images_dir = os.path.join(root_dir, split, 'images')
            labels_dir = os.path.join(root_dir, split, 'labels')
            
            if not os.path.exists(images_dir):
                continue
                
            for img_path in glob.glob(os.path.join(images_dir, '*.*')):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, base_name + '.txt')
                
                label = 0 # default good
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    label = 1 # anomaly
                    
                # For training in anomaly detection, we only want to train on normal images.
                # If there are not enough normal images, it might fail, but we adhere to the logic.
                if is_train and label == 1:
                    continue # Skip anomalous images during training
                    
                self.samples.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path
