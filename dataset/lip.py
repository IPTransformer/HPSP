import os
import torch
from torch.utils import data
from torchvision import transforms
import random  # caution: dif from np.random
import cv2
# import matplotlib.pyplot as plt
import numpy as np




class augmentation_scale(object):

    def __init__(self, scale_min=0.8, scale_max=1.2, istrain=True, crop_size=384.):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.is_train = istrain
        self.crop_size = crop_size


    def __call__(self, sample):

        img = sample['image']
        lbl = sample['label']

        dice = random.random()
        scale_multiplier = (self.scale_max - self.scale_min) * dice + self.scale_min
        long_size = max(img.shape[0], img.shape[1])
        base_scale = max(self.crop_size) / long_size

        if self.is_train:
            scale = base_scale * scale_multiplier
        else:
            scale = base_scale
        # 	scale = base_scale * scale_multiplier

        resized_im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        resized_catg = cv2.resize(lbl, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


        return {'image': resized_im,
                'label': resized_catg,
                }


class augmentation_cropped(object):

    def __init__(self, crop_x=384, crop_y=384, max_center_trans=40):
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.max_center_trans=max_center_trans

    def __call__(self, sample):

        im = sample['image']
        lbl = sample['label']



        dice_x = random.random()
        dice_y = random.random()

        x_offset = int((dice_x - 0.5) * 2 * self.max_center_trans)
        y_offset = int((dice_y - 0.5) * 2 * self.max_center_trans)

        h, w, c = im.shape
        cx=int(w/2)
        cy=int(h/2)

        # print x_offset, y_offset
        cx = np.clip(cx, 0, im.shape[1]-1)
        cy = np.clip(cy, 0, im.shape[0]-1)

        new_obj_center_x = cx + x_offset
        new_obj_center_y = cy + y_offset
        # print(crop_y,crop_x)
        cropped_im = np.zeros((self.crop_y, self.crop_x, 3), dtype='float') + 128.
        cropped_catg = np.ones((self.crop_y, self.crop_x), dtype='int')*255
        # cropped_human = np.ones((self.crop_y, self.crop_x), dtype='int')*255
        # cropped_im = np.zeros((crop_y, crop_x, 3), dtype="float")
        # cropped_im = 128*np.ones((crop_y, crop_x, 3), dtype="float")

        offset_start_x = int(new_obj_center_x - self.crop_x / 2.0)
        offset_start_y = int(new_obj_center_y - self.crop_y / 2.0)

        crop_start_x = max(offset_start_x, 0)
        crop_start_y = max(offset_start_y, 0)

        store_start_x = max(-offset_start_x, 0)
        store_start_y = max(-offset_start_y, 0)

        offset_end_x = int(new_obj_center_x + self.crop_x / 2.0)
        offset_end_y = int(new_obj_center_y + self.crop_y / 2.0)

        crop_end_x = min(offset_end_x, im.shape[1] - 1)
        crop_end_y = min(offset_end_y, im.shape[0] - 1)

        store_end_x = store_start_x + (crop_end_x - crop_start_x)
        store_end_y = store_start_y + (crop_end_y - crop_start_y)


            # (83, 536, 3) 20 1 0 0 83 90
        cropped_im[store_start_y:store_end_y, store_start_x:store_end_x, :] = im[crop_start_y:crop_end_y,
                                                                              crop_start_x:crop_end_x, :]
        cropped_catg[store_start_y:store_end_y, store_start_x:store_end_x] = lbl[crop_start_y:crop_end_y,
                                                                              crop_start_x:crop_end_x]


        return {'image': cropped_im.astype(np.uint8),
                'label': cropped_catg,
                }


class ResizeCrop(object):
    def __init__(self, h_resize=512, w_resize=512):
        self.h_resize = h_resize
        self.w_resize = w_resize

    def __call__(self, sample):
        img = sample['image']
        lbl = sample['label']

        # lbl_bw = copy.deepcopy(lbl)
        # lbl_bw[lbl_bw > 0] = 1
        # bbox = regionprops(lbl_bw)[0].bbox
        #
        # h_face = bbox[2] - bbox[0]
        # w_face = bbox[3] - bbox[1]
        h, w = img.shape[:2]
        h_resize = self.h_resize
        w_resize = self.w_resize

        rescale_ratio = min(float(w_resize) / w, float(h_resize) / h)

        # rescale
        h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(lbl, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

        # pad
        h_pad = max((h_resize - h_new), 0)
        w_pad = max((w_resize - w_new), 0)
        p1 = h_pad // 2
        p2 = h_pad - p1
        p3 = w_pad // 2
        p4 = w_pad - p3

        img = np.pad(img, ((p1, p2), (p3, p4), (0, 0)), 'constant', constant_values=(111, 111))
        mask = np.pad(mask, ((p1, p2), (p3, p4)), 'constant', constant_values=(255, 255))

        return {'image': img,
                'label': mask}



class Resize(object):
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        rescale_ratio = np.random.uniform(self.min_scale, self.max_scale)
        h, w, _ = img.shape

        # rescale
        h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

        return {'image': img,
                'label': mask}


class RandomCrop(object):
    def __init__(self, crop_size=(768, 768)):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        mask = self.crop(mask, crop_bbox)

        if img.shape[0] != self.crop_size[0] or img.shape[1] != self.crop_size[1]:
            t_size = self.crop_size[0]
            p1 = (t_size - img.shape[0]) // 2
            p2 = t_size - img.shape[0] - p1

            p3 = (t_size - img.shape[1]) // 2
            p4 = t_size - img.shape[1] - p3

            img = np.pad(img, ((p1, p2), (p3, p4), (0, 0)), 'constant', constant_values=(0, 0))
            mask = np.pad(mask, ((p1, p2), (p3, p4)), 'constant', constant_values=(255, 255))

        return {'image': img,
                'label': mask}


def exchange(mat, num1, num2):
    mat[mat == num1] = 100
    mat[mat == num2] = num1
    mat[mat == 100] = num2
    return mat


class FlipHorizon(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.randint(0, 1):
            img = np.ascontiguousarray(img[:, ::-1, ...])
            mask = np.ascontiguousarray(mask[:, ::-1, ...])

            mask = exchange(mask, 2, 3)
            mask = exchange(mask, 4, 5)
            mask = exchange(mask, 11, 12)
            mask = exchange(mask, 13, 14)

        return {'image': img,
                'label': mask}


class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) +
                            random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['image']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {'image': img,
                'label': results['label']}


class Pad(object):
    def __init__(self, pad_img=0, pad_mask=255):
        self.pad_img = pad_img
        self.pad_mask = pad_mask

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if img.shape[0] % 32 == 0 and img.shape[1] % 32 == 0:
            return {'image': img,
                    'label': mask}
        else:
            t_size = ((img.shape[0] - 1) // 32 + 1) * 32
            p1 = (t_size - img.shape[0]) // 2
            p2 = t_size - img.shape[0] - p1

            t_size = ((img.shape[1] - 1) // 32 + 1) * 32
            p3 = (t_size - img.shape[1]) // 2
            p4 = t_size - img.shape[1] - p3

            img = np.pad(img, ((p1, p2), (p3, p4), (0, 0)), 'constant', constant_values=(self.pad_img, self.pad_img))
            mask = np.pad(mask, ((p1, p2), (p3, p4)), 'constant', constant_values=(self.pad_mask, self.pad_mask))

            # print('pading', img.shape, mask.shape)

            return {'image': img,
                    'label': mask}


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0):
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        mean = np.array(self.mean, dtype=np.float32)
        mean *= self.max_pixel_value

        std = np.array(self.std, dtype=np.float32)
        std *= self.max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator

        return {'image': img,
                'label': mask}


class ToTensor(object):
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).long()

        return {'image': img,
                'catg': mask}


class LIPDataset(data.Dataset):
    NUM_CLASSES = 20

    def __init__(self, args, data_path='/mnt/workspace/caoleilei/models/smp_lapa/datas/LIP/',
                 mode="train",
                 img_size=512):

        self.data_path = data_path
        self.args = args
        self.mode = mode

        self.img_size = img_size

        self.img_dir = self.data_path + mode + '/images/'
        self.img_names = os.listdir(self.img_dir)

        if args.local_rank == 0:
            print('%s dataset: %d images' % (self.mode, len(self.img_names)))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            index, _ = item
        else:
            index, _ = item, 256
        # img_size = 512
        img_name = self.img_names[index]

        if self.mode == 'train':
            img_path = os.path.join(self.data_path + 'train/images', img_name)
        elif self.mode == 'val':
            img_path = os.path.join(self.data_path + 'val/images', img_name)

        lbl_path = img_path.replace('.jpg', '.png')
        lbl_path = lbl_path.replace('images', 'labels')

        _img = cv2.imread(img_path)
        bgr = _img[:, :, :3]
        _img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        _tmp = cv2.imread(lbl_path)
        _tmp = _tmp[:, :, 0]

        sample = {'image': _img, 'label': _tmp}
        if self.mode == 'train' or self.mode == 'finetune':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'swa':
            return self.transform_tr(sample)  # ['image']

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # Resize(min_scale=0.8, max_scale=1.2),
            # RandomCrop(crop_size=(768, 768)),
            augmentation_scale(crop_size=[self.img_size, self.img_size]),
            augmentation_cropped(crop_x=self.img_size, crop_y=self.img_size),
            # ResizeCrop(self.img_size, self.img_size),
            # FlipHorizon(),
            PhotoMetricDistortion(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            ResizeCrop(self.img_size, self.img_size),
            # Pad(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            Pad(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])
        return composed_transforms(sample)


def vis_img(image):
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image *= np.array([0.229, 0.224, 0.225])
    image += np.array([0.485, 0.456, 0.406])
    image *= 255
    return image.astype(np.int)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")

    args = parser.parse_args()

    args.local_rank = 0

    dataset = LIPDataset(args, mode='train')
    print(len(dataset))
    for i in range(len(dataset)):

        sample = dataset[i]
        image = sample['image']
        label = sample['label']

        fig = plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(vis_img(image))
        plt.subplot(122)
        label[label==255]=0
        plt.imshow(label)
        plt.show()

        print(image.shape, label.shape)

