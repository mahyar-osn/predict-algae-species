from torch.utils.data import Dataset
import cv2


class SegmentationDataSet(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # grab the image path from the current index

        """ load the image from disk, swap its channels from BGR to RGB,
        and read the associated mask from disk in grayscale mode """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotations = cv2.imread(self.mask_paths[idx], 0)

        """ check to see if we are applying any transformations """
        if self.transforms is not None:
            image = self.transforms(image)
            annotations = self.transforms(annotations)

        return image, annotations
