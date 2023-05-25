import os
from glob import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from region_manipulations import RegionManipulation


class ForgeryNet_Base(Dataset):
    def __init__(
        self,
        train_images=None,
        augmented_images=None,
        test_images=None,
        image_size=(256, 256),
        mode="train",
        manipulation_type="6transforms",
        deepfake_type=None,
        stage1=True,
        data_augmentation_type="basic",
    ):
        """
        Dataset object
        ForgeryNet base dataset

        :param train_images[str]: folder to training images
        :param augmented_images [str]: folder of offline generated spliced images
        :param test_images[str]: folder to test dataset whose structure should be compatible with datasets.ImageFolder
        :param image_size[tuple]: image size for training
        :param mode[str]: options ['train', 'test']
        :param manipulation_type[str]: 6transforms is default as 6 transformations can be applied to an input image
        :param deepfake_type[str]: forgery folder name, options ['cdf','ff++_all', 'fgnet', 'stylegan2', 'stargan2',...]
        :param stage1[bool]: if True, specifies that the dataset is used to train stage_one
        :param data_augmentation_type[str]: specifies how to augmented training images options ['none', 'basic']
        """

        self.deepfake_type = deepfake_type
        self.mode = mode
        self.train_images = train_images
        self.augmented_images = augmented_images
        self.test_images = test_images
        self.crop_size = image_size
        self.data_augmentation_type = data_augmentation_type

        if self.train_images and mode == "train" and stage1:

            self.region_transform = RegionManipulation(
                type=manipulation_type,
                generate_online=False,  # only offline manipulations are supported for now
                augmented_images=glob(self.augmented_images + "/*.png"),
                real_data_paths=glob(self.train_images + "/*.png"),
            )
        else:
            self.region_transform = None

        if type(image_size) is not tuple:
            image_size = (image_size, image_size)

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if self.data_augmentation_type == "basic":
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.3),
                    transforms.RandomCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.train_transform = self.test_transform

        if self.train_images and self.mode == "train":
            self.images = glob(self.train_images + "/*.png")

        elif self.test_images and self.mode == "test":
            assert self.deepfake_type is not None
            self.test_images = os.path.join(self.test_images, self.deepfake_type)
            self.images = datasets.ImageFolder(
                self.test_images, transform=self.test_transform
            )

        elif self.train_images and self.mode == "test":
            self.images = datasets.ImageFolder(
                self.train_images, transform=self.test_transform
            )

        else:
            raise ValueError("No dataset path nor a mode have been provided.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        :return:  if self.mode is 'train' then a torch tensor is returned else a
        torch tensor and its associated label
        """
        if self.mode == "train":
            image_path = self.images[item]
            image = Image.open(image_path).convert("RGB")
            if self.region_transform is not None:
                output_img = self.region_transform(
                    image=image,
                    img_filename=image_path,
                )
            else:
                output_img = [image]
            transformed = [self.train_transform(i) for i in output_img]
            return transformed

        else:
            image, label = self.images.samples[item]
            image = Image.open(image).convert("RGB")
            image = self.test_transform(image)
            return image, label


class ForgeryNet(ForgeryNet_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.mode == "test":
            # 1 is real, 0 is fake
            self.images.samples = [
                (d, 1) if i == self.images.class_to_idx["Test_real_faces"] else (d, 0)
                for d, i in self.images.samples
            ]

        else:
            self.images = glob(self.train_images + "/*.png")
