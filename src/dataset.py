from PIL import Image
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        
        for label, class_name in enumerate(classes):
            class_dir = root_dir / class_name

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((image_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)
        
        return image, label