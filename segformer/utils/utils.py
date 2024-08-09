import torchvision.transforms.functional as TF

class InvertMaskColors(object):
    def __call__(self, mask):
        if mask.mode != 'L':
            raise ValueError("Input mask should be in grayscale (L) mode")
        mask = TF.to_tensor(mask)
        mask = 1.0 - mask
        return TF.to_pil_image(mask)
