import torch
from torch import Tensor
from torchvision.transforms import functional as F


class SquareVideo(object):
    """
    cuts the video clip to a square shape by cutting the longer side
    Takes and returns a video clip of shape (T, C, H, W)
    """

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be squared. Size is (T, C, H, W)
        Return:
            clip (torch.tensor): video clip of shape (T, C, H, H)
        """
        h, w = clip.shape[2], clip.shape[3]
        if h > w:
            margin = (h - w) // 2
            clip = clip[:, :, margin : margin + w, :]
        elif w > h:
            margin = (w - h) // 2
            clip = clip[:, :, :, margin : margin + h]
        return clip


class ResizeVideo(object):
    """
    Resize the video clip to the desired size
    Args:
        video_size (2-tuple): desired heigth and width of the video
        interpolation_mode (InterpolationMode): interpolation mode for resizing
    """

    def __init__(self, video_size, interpolation_mode):
        self.video_size = video_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be resized. Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        clip = F.resize(
            clip,
            (self.video_size[0], self.video_size[1]),
            interpolation=self.interpolation_mode,
        )
        return clip

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(video_size={0}, interpolation_mode={1})".format(
                self.video_size, self.interpolation_mode
            )
        )


class ResizeVideo2(object):
    """
    Resize the video clip to the desired size
    Args:
        video_size (2-tuple): desired heigth and width of the video
        interpolation_mode (InterpolationMode): interpolation mode for resizing
    """

    def __init__(self, video_size, interpolation_mode):
        self.video_size = video_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be resized. Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        clip = F.resize(
            clip,
            (self.video_size,),
            interpolation=self.interpolation_mode,
        )
        return clip

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(video_size={0}, interpolation_mode={1})".format(
                self.video_size, self.interpolation_mode
            )
        )


class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """

    def __init__(self, max_pixel_value=255.0):
        self.max_pixel_value = max_pixel_value

    def __call__(self, clip: Tensor):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return (clip / self.max_pixel_value).to(torch.float32)

    def __repr__(self):
        return self.__class__.__name__


class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, inplace={2})".format(
            self.mean, self.std, self.inplace
        )


class TemporalJittering(object):
    """
    Temporally jitter the video clip by randomly sampling a start index

    Args:
        vid_length (int): padding length / final length of the video clip
    """

    def __init__(self, vid_length) -> None:
        self.vid_length = vid_length

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be temporally jittered. Size is (T, ...)
        Return:
            clip (torch.tensor): Size is (T, ...)
        """
        clip_len = clip.shape[0]
        if clip_len <= self.vid_length:
            return clip
        start_idx = torch.randint(0, clip_len - self.vid_length, (1,))
        return clip[start_idx:]
