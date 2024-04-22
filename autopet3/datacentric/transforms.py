import math
import random
from collections.abc import Hashable
from typing import Dict, List, Tuple

import monai.transforms as mt
import torch

from autopet3.datacentric.utils import get_file_dict_nn, read_split


class CustomSampleCropd(mt.Transform):
    def __init__(
        self,
        keys: List[str],
        label_key: str,
        roi_size: Tuple,
        border_pad: Tuple[int, int, int] = (64, 80, 56),
        return_pad: bool = False,
        prob: float = 0.5,
    ):
        """Custom sampling class for sampling foreground and background samples.
        Args:
            keys (List[str]): List of keys.
            label_key (str): Key for labels.
            roi_size (Tuple[int, int, int]): Size of the ROI.
            border_pad (Tuple[int, int, int], optional): Padding for borders. Defaults to (64, 80, 56).
            return_pad (bool, optional): Flag to return padding. Defaults to False.
            prob (float, optional): Probability. Defaults to 0.5.

        """
        self.keys = keys
        self.label_key = label_key
        self.roi_size = roi_size
        self.prob = prob
        self.return_pad = return_pad
        self.crop_content = mt.CropForegroundd(keys=self.keys, source_key="ct", allow_smaller=True)
        self.pad = mt.BorderPadd(keys=self.keys, spatial_border=border_pad)
        self.rand_crop = mt.RandSpatialCropd(keys=self.keys, roi_size=self.roi_size)
        self.foreground_crop = mt.RandCropByPosNegLabeld(
            keys=self.keys, label_key=self.label_key, spatial_size=self.roi_size, pos=1.0, neg=0.0, num_samples=1
        )

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        label = data[self.label_key]
        data = self.crop_content(data)
        data = self.pad(data)

        if self.return_pad:
            return data
        if torch.any(label) and random.random() > self.prob:
            return self.foreground_crop(data)[0]
        return self.rand_crop(data)


class Unpackd(mt.Transform):
    def __init__(self, keys: List[str]):
        """Unpackd is a transform that extracts specified keys from a dictionary and returns them as a tuple.
        Args:
           keys (List[str]): The keys to extract from the dictionary.
        Returns:
           tuple: A tuple containing the values corresponding to the specified keys.

        """
        self.keys = keys

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> tuple:
        return tuple([data[k] for k in self.keys])


def get_transforms(
    stage: str,
    target_shape: Tuple,
    resample: bool = False,
    load: bool = True,
    spacing: Tuple[float, float, float] = (2.0364201068878174, 2.0364201068878174, 3.0),
    ct_percentiles: Tuple[float, float] = (-831.9483642578125, 1127.5189013671843),
    pet_percentiles: Tuple[float, float] = (1.0438873767852783, 51.595245361328125),
    pet_norm: torch.Tensor = torch.Tensor((7.089671984867433, 8.01080984806995)),
    ct_norm: torch.Tensor = torch.Tensor((107.77672799269146, 286.1725548247846)),
) -> mt.Compose:
    """The get_transforms function generates a series of transforms based on the stage and target shape.
    It performs loading, resampling, shifting intensity ranges, and applying various augmentation techniques such
    as affine transforms, noise, blur, intensity adjustments, flips, and more. It also performs normalization and
    concatenates the transformed data into a single tensor.
    Args:
        stage (str): The stage of the transformation (e.g., "train", "val", "val2", "val_sampled").
        target_shape (Tuple[int, int, int]): The target shape of the transformation.
        resample (bool): Flag indicating whether resampling should be performed.
        load (bool): Flag indicating whether loading should be performed.
        spacing (Tuple[float, float, float]): The spacing for resampling.
        ct_percentiles (Tuple[float, float]): The percentiles for CT normalization.
        pet_percentiles (Tuple[float, float]): The percentiles for PET normalization.
        pet_norm (torch.Tensor): The normalization factor for PET.
        ct_norm (torch.Tensor): The normalization factor for CT.
    Returns:
        mt.Compose: A composed set of transforms.

    """
    input_keys = ["ct", "pet"]
    if stage in ["train", "val", "val2", "val_sampled"]:
        keys = ["ct", "pet", "label"]
        out = ["image", "label"]
        mode = ("bilinear", "bilinear", "nearest")
    else:
        keys = ["ct", "pet"]
        out = ["image"]
        mode = ("bilinear", "bilinear")

    # Define a list to store all the transforms
    all_transforms = []

    # loading
    if load:
        load_transforms = [
            mt.LoadImaged(keys=keys),
            mt.EnsureChannelFirstd(keys=keys),
            mt.EnsureTyped(keys=keys),
        ]
        all_transforms.extend(load_transforms)

    if resample:
        # resampling
        sample_transforms = [
            mt.Orientationd(keys=keys, axcodes="LAS"),  # RAS
            mt.Spacingd(
                keys=keys,
                pixdim=spacing,
                mode=mode,
            ),
        ]
        all_transforms.extend(sample_transforms)

    # shift ct and pet in value range 0 to 1 for transforms to work properly eg zero padding
    shift_transforms = [
        mt.ScaleIntensityRanged(
            keys=["ct"], a_min=ct_percentiles[0], a_max=ct_percentiles[1], b_min=0, b_max=1, clip=True
        ),
        mt.ScaleIntensityRanged(
            keys=["pet"], a_min=pet_percentiles[0], a_max=pet_percentiles[1], b_min=0, b_max=1, clip=True
        ),
    ]
    all_transforms.extend(shift_transforms)

    if stage == "train":
        other_transforms = [
            # pad to target shape times 1.2 times sqrt(2) (because of affine transforms) and sample 50% a class
            # foreground part, after that apply affine transform and crop
            # mt.SpatialPadd(keys=keys, spatial_size=tuple(int(math.sqrt(2) * x * 1.2) for x in target_shape)),
            CustomSampleCropd(
                keys=keys, label_key="label", roi_size=tuple(int(math.sqrt(2) * x * 1.2) for x in target_shape)
            ),
            mt.RandAffined(
                keys=keys,
                mode=mode,
                prob=0.2,
                spatial_size=target_shape,
                translate_range=(20, 20, 20),
                rotate_range=(0.52, 0.52, 0.52),
                scale_range=((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)),
                padding_mode="zeros",
            ),
            # noise, blur, intensity and flips
            mt.RandGaussianNoised(keys=input_keys, std=0.1, prob=0.15),
            mt.RandGaussianSmoothd(
                keys=input_keys,
                sigma_x=(0.5, 1),
                sigma_y=(0.5, 1),
                sigma_z=(0.5, 1),
                prob=0.2,
            ),
            mt.RandScaleIntensityd(keys=input_keys, factors=0.25, prob=0.15),
            mt.RandSimulateLowResolutiond(keys=input_keys, zoom_range=(0.5, 1), prob=0.25),
            mt.RandAdjustContrastd(keys=input_keys, gamma=(0.7, 1.5), invert_image=True, retain_stats=True, prob=0.1),
            mt.RandAdjustContrastd(keys=input_keys, gamma=(0.7, 1.5), invert_image=False, retain_stats=True, prob=0.3),
            mt.RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
            mt.RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
            mt.RandFlipd(keys=keys, spatial_axis=[2], prob=0.5),
        ]
        all_transforms.extend(other_transforms)

    elif stage == "val_sampled":
        other_transforms = [
            mt.SpatialPadd(keys=keys, spatial_size=target_shape),
            CustomSampleCropd(keys=keys, label_key="label", roi_size=target_shape),
        ]
        all_transforms.extend(other_transforms)

    elif stage == "val":
        other_transforms = [
            mt.SpatialPadd(keys=keys, spatial_size=target_shape),
        ]
        all_transforms.extend(other_transforms)

    normalization_transforms = [
        # shift ct and pet range back to original and clip (again)
        mt.ScaleIntensityRanged(
            keys=["ct"], a_min=0, a_max=1, b_min=ct_percentiles[0], b_max=ct_percentiles[1], clip=True
        ),
        mt.ScaleIntensityRanged(
            keys=["pet"], a_min=0, a_max=1, b_min=pet_percentiles[0], b_max=pet_percentiles[1], clip=True
        ),
        mt.NormalizeIntensityd(keys=["ct"], subtrahend=ct_norm[0], divisor=ct_norm[1]),
        mt.NormalizeIntensityd(keys=["pet"], subtrahend=pet_norm[0], divisor=pet_norm[1]),
    ]
    all_transforms.extend(normalization_transforms)

    final_transforms = [
        mt.ConcatItemsd(keys=input_keys, name="image", dim=0),
        mt.EnsureTyped(keys=keys),
        mt.ToTensord(keys=keys),
        Unpackd(keys=out),  # unpack dict, pickable version
    ]
    all_transforms.extend(final_transforms)
    return mt.Compose(all_transforms)


if __name__ == "__main__":
    # Test the transforms
    from autopet3.fixed.utils import plot_ct_pet_label

    split = read_split("../../test/data/splits_final.json", 0)
    test_sample = get_file_dict_nn("../../test/data/", split["train"], suffix=".nii.gz")[0]
    print(test_sample)

    transform = get_transforms(stage="train", resample=True, target_shape=(112, 160, 128))
    res = transform(test_sample)
    plot_ct_pet_label(ct=res[0][0], pet=res[0][1], label=res[1])

    transform = get_transforms(stage="val_sampled", resample=False, target_shape=(112, 160, 128))
    res = transform(test_sample)
    plot_ct_pet_label(ct=res[0][0], pet=res[0][1], label=res[1])
