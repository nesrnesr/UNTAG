import os
import re

from PIL import Image
from torchvision import transforms

# https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

FACIAL_REGIONS_LMKS_IDS = {
    "brows": [9, 68, 156, 124, 53, 52, 8, 282, 283, 353, 333, 298],
    "eyes": [8, 222, 224, 35, 230, 6, 450, 265, 445, 442],
    "nose": [193, 203, 164, 423, 417],
    "mouth": [164, 165, 212, 200, 432, 391],
    "background": [108, 68, 143, 213, 210, 208, 426, 430, 433, 372, 298, 337],
}


class RegionManipulation(object):
    def __init__(
        self,
        transform=True,
        type="6transforms",
        generate_online=False,
        augmented_images=None,
        real_data_paths=None,
    ):
        """
        RegionManipulation creates data augmentations by facial region splicing in an offline manner for instance.

        :transform[Bool]: - if True uses Color Jitter
        :type[str]: 6transforms
        :generate_online: [Bool]: - if True generates transformations on the go
        :augmented_images: Union[list[str], None]: - list of all augmented images already generated in an offline manner.
        :real_data_paths: Union[list[str], None]: - list of real training sample paths.
        """

        self.type = type
        self.generate_online = generate_online

        if not self.generate_online:
            assert augmented_images is not None
            self.augmented_images = augmented_images

        assert real_data_paths is not None
        self.real_data_paths = real_data_paths

        if transform:
            self.transform = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            )
        else:
            self.transform = None

    def forge_region(self, image, region):
        """
        Forges a given region forgery

        :param image: [PIL] - original image
        :param region: [str] - region to replace

        :return: PIL image after region manipulation
        """
        pass

    def __call__(self, image=None, img_filename=None):
        """

        :param image: [PIL] - original image
        :param img_filename [str] - img filename

        :return[Tuple]: original image, its manipulated variants.
        """

        if self.generate_online:
            assert image is not None

            if self.type == "6transforms":
                spliced_images = []
                for region in FACIAL_REGIONS_LMKS_IDS:
                    spliced_images.append(self.forge_region(image, region))
                return image, *spliced_images

        else:
            assert img_filename is not None

            # images have names as: 421_00000.png
            # extract the idx from img_filename
            img_idx = re.search("^[0-9]+", os.path.basename(img_filename)).group(0)

            # find all augmentations of the current image
            augmented_imgs = [
                img_path
                for img_path in self.augmented_images
                if os.path.basename(img_path).find(img_idx + "_") == 0
            ]

            # augmented images are named as "999_10000.png"
            if self.type == "6transforms":
                # regions from filename: bg, mouth, nose, eyes, brows
                single_regions_labels = ["00001", "00010", "00100", "01000", "10000"]
                augmented_images = []
                for region in single_regions_labels:
                    region_img = [
                        img_path for img_path in augmented_imgs if region in img_path
                    ]
                    augmented_images.append(Image.open(region_img[0]))
                return Image.open(img_filename), *augmented_images
