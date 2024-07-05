from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])
        self.augment_bg = Compose([
            Resize(self.size),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        # print("LABELS + BOXES[0]")
        # print(labels)
        # print(boxes[0].shape)
        # from torchvision.transforms.functional import to_pil_image
        # if len(boxes) == 0:
        # to_pil_image(img).save('./debug_image.png')
        if len(boxes) > 0:
            return self.augment(img, boxes, labels)
        else:
            # print("No bounding boxes to augment; augmenting background image.")
            return self.augment_bg(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])
        self.transform_bg = Compose([
            Resize(size),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        if len(boxes) > 0:
            return self.transform(image, boxes, labels)
        else:
            # print("No bounding boxes to augment; augmenting background image.")
            return self.transform_bg(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
