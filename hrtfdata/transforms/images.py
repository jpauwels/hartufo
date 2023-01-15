from typing import Optional
from PIL.Image import Image, LANCZOS


class CenterCropResizeGreyscale:
    def __init__(self, resize_width: Optional[int], resize_height: Optional[int], crop_width: Optional[int]=None, crop_height: Optional[int]=None):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.crop_width = crop_width
        self.crop_height = crop_height


    def __call__(self, image: Image) -> Image:
        orig_width, orig_height = image.size
        if self.resize_width is None:
            resize_width = orig_width if self.crop_width is None else self.crop_width
        else:
            resize_width = self.resize_width
        if self.resize_height is None:
            resize_height = orig_height if self.crop_height is None else self.crop_height
        else:
            resize_height = self.resize_height
        if self.crop_width is not None or self.crop_height is not None:
            if self.crop_width is not None:
                box_left = (orig_width - self.crop_width) // 2
                box_right = box_left + self.crop_width
            else:
                box_left = 0
                box_right = orig_width
            if self.crop_height is not None:
                box_top = (orig_height - self.crop_height) // 2
                box_bottom = box_top + self.crop_height
            else:
                box_top = 0
                box_bottom = orig_height
            box = (box_left, box_top, box_right, box_bottom)
        else:
            box = None
        return image.resize((resize_width, resize_height), resample=LANCZOS, box=box).convert('L')
