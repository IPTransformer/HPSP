import os
import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
import random  # caution: dif from np.random
import cv2
# import matplotlib.pyplot as plt
import numpy as np

class ResizeCrop(object):
    def __init__(self, h_resize=512, w_resize=512):
        self.h_resize = h_resize
        self.w_resize = w_resize

    def __call__(self, sample):

        im = sample['image']
        catg = sample['catg']
        joints = sample['joint']
        person_center = sample['person_center']

        h, w = im.shape[:2]
        h_resize = self.h_resize
        w_resize = self.w_resize

        rescale_ratio = min(float(w_resize) / w, float(h_resize) / h)

        # rescale
        h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
        img = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(catg, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

        # pad
        h_pad = max((h_resize - h_new), 0)
        w_pad = max((w_resize - w_new), 0)
        p1 = h_pad // 2
        p2 = h_pad - p1
        p3 = w_pad // 2
        p4 = w_pad - p3

        img = np.pad(img, ((p1, p2), (p3, p4), (0, 0)), 'constant', constant_values=(128, 128))
        mask = np.pad(mask, ((p1, p2), (p3, p4)), 'constant', constant_values=(255, 255))
        new_joints = joints[:,:2]*rescale_ratio + np.array([p3, p1])
        person_center = person_center*rescale_ratio + np.array([p3, p1])

        joints[:,:2] = new_joints

        return {'image': img,
                'catg': mask,
                'joint': joints,
                'person_center': person_center}


class augmentation_scale(object):

    def __init__(self, scale_min=0.8, scale_max=1.5, istrain=True, crop_size=384.):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.is_train = istrain
        self.crop_size = crop_size


    def __call__(self, sample):

        im = sample['image']
        catg = sample['catg']
        human = sample['human']
        person_center = sample['person_center']


        dice = random.random()
        scale_multiplier = (self.scale_max - self.scale_min) * dice + self.scale_min
        long_size = max(im.shape[0], im.shape[1])
        base_scale = max(self.crop_size) / long_size

        if self.is_train:
            scale = base_scale * scale_multiplier
        else:
            scale = base_scale
        # 	scale = base_scale * scale_multiplier

        resized_im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        resized_catg = cv2.resize(catg, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        resized_human = cv2.resize(human, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        resized_center = person_center*scale



        return {'image': resized_im,
                'catg': resized_catg,
                'human': resized_human,
                'person_center': resized_center}

class augmentation_rotate(object):

    def __init__(self, max_rotate_degree=40):
        self.max_rotate_degree = max_rotate_degree

    def __call__(self, sample):

        im = sample['image']
        catg = sample['catg']
        human = sample['human']
        person_center = sample['person_center']

        dice = random.random()
        degree = (dice - 0.5) * 2 * self.max_rotate_degree

        im_width = im.shape[1]
        im_height = im.shape[0]
        M = cv2.getRotationMatrix2D(center=(im_width / 2, im_height / 2), angle=degree, scale=1)
        r = np.deg2rad(degree)
        new_im_width = abs(np.sin(r) * im_height) + abs(np.cos(r) * im_width)
        new_im_height = abs(np.sin(r) * im_width) + abs(np.cos(r) * im_height)
        tx = (new_im_width - im_width) / 2
        ty = (new_im_height - im_height) / 2
        M[0, 2] += tx
        M[1, 2] += ty

        rotated_im = cv2.warpAffine(im, M, dsize=(int(new_im_width), int(new_im_height)),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(128, 128, 128))

        rotated_catg = cv2.warpAffine(catg, M, dsize=(int(new_im_width), int(new_im_height)),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255))

        rotated_human = cv2.warpAffine(human, M, dsize=(int(new_im_width), int(new_im_height)),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255))

        center = person_center
        center_trans = center.transpose()
        center_padded = np.ones((3, 1))
        center_padded[0:2, :] = center_trans
        rotated_center = np.dot(M, center_padded)

        rotated_center = rotated_center.transpose()


        return {'image': rotated_im,
                'catg': rotated_catg,
                'human': rotated_human,
                'person_center': rotated_center}

class augmentation_cropped(object):

    def __init__(self, crop_x=384, crop_y=384, max_center_trans=40):
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.max_center_trans=max_center_trans

    def __call__(self, sample):

        im = sample['image']
        catg = sample['catg']
        human = sample['human']
        obj_center = sample['person_center']

        dice_x = random.random()
        dice_y = random.random()

        x_offset = int((dice_x - 0.5) * 2 * self.max_center_trans)
        y_offset = int((dice_y - 0.5) * 2 * self.max_center_trans)

        # print x_offset, y_offset
        obj_center[0, 0] = np.clip(obj_center[0, 0], 0, im.shape[1]-1)
        obj_center[0, 1] = np.clip(obj_center[0, 1], 0, im.shape[0]-1)

        new_obj_center_x = obj_center[0, 0] + x_offset
        new_obj_center_y = obj_center[0, 1] + y_offset
        # print(crop_y,crop_x)
        cropped_im = np.zeros((self.crop_y, self.crop_x, 3), dtype='float') + 128.
        cropped_catg = np.ones((self.crop_y, self.crop_x), dtype='int')*255
        cropped_human = np.ones((self.crop_y, self.crop_x), dtype='int')*255
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

        if cropped_im[store_start_y:store_end_y, store_start_x:store_end_x, :].shape != im[crop_start_y:crop_end_y,
                                                                              crop_start_x:crop_end_x, :].shape:
            print(cropped_im[store_start_y:store_end_y, store_start_x:store_end_x, :].shape)
            print(im[crop_start_y:crop_end_y,crop_start_x:crop_end_x, :].shape, obj_center)
            print(im.shape, x_offset, y_offset, store_start_x, store_start_y, crop_start_y, crop_start_x)
            # (83, 536, 3) 20 1 0 0 83 90
        cropped_im[store_start_y:store_end_y, store_start_x:store_end_x, :] = im[crop_start_y:crop_end_y,
                                                                              crop_start_x:crop_end_x, :]
        cropped_catg[store_start_y:store_end_y, store_start_x:store_end_x] = catg[crop_start_y:crop_end_y,
                                                                              crop_start_x:crop_end_x]

        cropped_human[store_start_y:store_end_y, store_start_x:store_end_x] = human[crop_start_y:crop_end_y,
                                                                              crop_start_x:crop_end_x]



        cropped_center = obj_center[:, :2] - np.array([[crop_start_x, crop_start_y]]) \
                            + np.array([[store_start_x, store_start_y]])



        return {'image': cropped_im,
                'catg': cropped_catg,
                'human': cropped_human,
                'person_center': cropped_center}

class augmentation_flip(object):

    def __init__(self, flip_prob=0.5):
        self.flip_prob=flip_prob


    def flip_human(self, img):
        img_before = img.copy()
        img_before[img_before!=255] = 0
        img[img==255] = 0
        num_human_plus1 = img.max() + 1
        img = num_human_plus1 - img
        img[img==num_human_plus1] = 0


        return img+img_before




    def __call__(self, sample):

        im = sample['image']
        catg = sample['catg']
        human = sample['human']
        obj_center = sample['person_center']

        dice = random.random()

        doflip = False
        if dice <= self.flip_prob:
            doflip = True

        if doflip:
            flipped_im = cv2.flip(im, 1)
            flipped_catg = cv2.flip(catg, 1)
            flipped_human = cv2.flip(human, 1)

            # flip mask for arm, leg, shoe
            right_idx = [14, 16, 18]
            left_idx = [15, 17, 19]
            for i in range(0, 3):
                right_pos = np.where(flipped_catg == right_idx[i])
                left_pos = np.where(flipped_catg == left_idx[i])
                flipped_catg[right_pos[0], right_pos[1]] = left_idx[i]
                flipped_catg[left_pos[0], left_pos[1]] = right_idx[i]
            # flip jointes
            _, imw, _ = flipped_im.shape
            flipped_center = obj_center.copy()
            flipped_center[:,0] = imw - 1 - flipped_center[:, 0]

            # flip human
            # flipped_human = human.copy()

        else:
            flipped_im = im.copy()
            flipped_catg = catg.copy()
            flipped_human = human.copy()
            flipped_center = obj_center.copy()

        return {'image': flipped_im,
                'catg': flipped_catg,
                'human': flipped_human,
                'person_center': flipped_center}

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

    def __call__(self, sample):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """


        img = sample['image']
        img = cv2.cvtColor(img.astype('int'), cv2.COLOR_RGB2BGR)
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

        # a = dict()
        sample.update({'image': img})

        return sample


class Pad(object):
    def __init__(self, pad_img=0, pad_mask=255):
        self.pad_img = pad_img
        self.pad_mask = pad_mask

    def __call__(self, sample):
        img = sample['image']
        mask = sample['catg']

        if img.shape[0] % 32 == 0 and img.shape[1] % 32 == 0:
            return {'image': img,
                    'catg': mask}
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
                    'catg': mask}


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0):
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def __call__(self, sample):
        img = sample['image']
        mean = np.array(self.mean, dtype=np.float32)
        mean *= self.max_pixel_value

        std = np.array(self.std, dtype=np.float32)
        std *= self.max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator

        sample.update({'image': img})


        return sample


class ToTensor(object):
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = sample['image']
        catg = sample['catg']
        human = sample['human']
        person_center = sample['person_center']

        img = torch.from_numpy(im.transpose(2, 0, 1))
        mask = torch.from_numpy(catg).long()
        human = torch.from_numpy(human).long()

        return {'image': img,
                'catg': mask,
                'person_center': person_center,
                'human': human,
                }


def get_center(img):
    img = img.cpu().numpy()

    c, h, w = img.shape
    o1, o2, o3 = np.where(img==1)
    ind_ = np.ones(c) * 999

    o1_ind = np.unique(o1)

    for i in o1_ind:
        ind_select = np.where(img[i]> 0)
        if len(ind_select[0])== 0:
            print(np.unique(o1))
        # print(ind_select[0].shape,ind_select[1].shape)

        x_min, y_min = ind_select[1].min(), ind_select[0].min()
        x_max, y_max = ind_select[1].max(), ind_select[0].max()

        cx = (x_min + x_max) / 2
        ind_[i] = cx
    # print(ind_)
    correct_img = img[np.argsort(ind_)]
    correct_img = correct_img.copy()
    bg = np.zeros((1, h, w))
    img_withbg = np.concatenate((bg, correct_img), axis=0)
    return correct_img, img_withbg


class ToOneHotTensor(object):
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = sample['image']
        catg = sample['catg']
        human = sample['human']
        person_center = sample['person_center']

        img = torch.from_numpy(im.transpose(2, 0, 1))
        mask = torch.from_numpy(catg).long()
        human = torch.from_numpy(human).long()
        human[human==255] = 0
        human = F.one_hot(human, num_classes=13)[:,:,1:]
        human = human.permute(2,0,1)


        human_correct, human_correct_withbg = get_center(human)
        human_correct = torch.from_numpy(human_correct)
        human_correct_withbg = torch.from_numpy(human_correct_withbg)
        humanmask_correct = human_correct_withbg.max(0)[1]


        return {'image': img,
                'catg': mask,
                'person_center': person_center,
                # 'human': human,
                'human_correct': human_correct,
                'humanmask_correct': humanmask_correct
                }



class CIHPDataset(data.Dataset):
    NUM_CLASSES = 20

    def __init__(self, args, data_path='datas/CIHP/',
                 mode="train",
                 img_size=384):

        self.data_path = data_path
        self.args = args
        self.mode = mode

        self.img_size = img_size

        if self.mode == 'train':
            self.img_dir = self.data_path+'/Training/Images'
        elif self.mode == 'val':
            self.img_dir = self.data_path+'Validation/Images'

        self.img_names = os.listdir(self.img_dir)
        # self.files = self.find_tif(self.mode)

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
        # print(img_name)
        if self.mode == 'train':
            img_path = os.path.join(self.data_path+'/Training/Images', img_name)
        elif self.mode == 'val':
            img_path = os.path.join(self.data_path+'Validation/Images', img_name)
        cat_path = img_path.replace('.jpg', '.png')
        cat_path = cat_path.replace('Images', 'Category_ids')

        human_path = img_path.replace('.jpg', '.png')
        human_path = human_path.replace('Images', 'Human_ids')

        _img = cv2.imread(img_path)
        bgr = _img[:, :, :3]
        _img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        _cat = cv2.imread(cat_path)
        _cat = _cat[:, :, 0]


        _human = cv2.imread(human_path)
        _human = _human[:, :, 0]

        H, W, C = _img.shape
        person_center = np.array([[W/2., H/2.]])

        sample = {'image': _img, 'catg': _cat, 'human': _human, 'person_center': person_center}
        if self.mode == 'train' or self.mode == 'finetune':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'swa':
            return self.transform_tr(sample)  # ['image']


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # PhotoMetricDistortion(),
            augmentation_scale(crop_size=self.img_size),
            augmentation_rotate(max_rotate_degree=self.args.rotate),
            augmentation_cropped(crop_x=self.img_size[1], crop_y=self.img_size[0], max_center_trans=40),
            augmentation_flip(flip_prob=0.5),
            # augmentation_scale(crop_size=self.img_size, istrain=False),
            # augmentation_rotate(max_rotate_degree=0),
            # augmentation_cropped(crop_x=self.img_size[1], crop_y=self.img_size[0], max_center_trans=0),
            # augmentation_flip(flip_prob=0.),
            # gen_edge(),
            # gen_heatmaps(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToOneHotTensor(),
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # ResizeCrop(self.img_size, self.img_size),
            augmentation_scale(istrain=False, crop_size=self.img_size),
            augmentation_cropped(max_center_trans=0, crop_x=self.img_size[1], crop_y=self.img_size[0]),
            # gen_edge(),
            # gen_heatmaps(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensor(),
            ToOneHotTensor(),
        ])
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

    dataset = CIHPDataset(args, mode='train', img_size=[512, 512])
    print(len(dataset))


    for i in range(len(dataset)):
    # for i in range(10):
    #     ind = random.randint(0, len(dataset))
        ind = i
        sample = dataset[ind]
        image = sample['image']
        catg = sample['catg']
        human = sample['humanmask_correct']
        human_ins = sample['human_correct']


        plt.imshow(vis_img(image))
        plt.show()
        # human[human==255] = 0
        plt.imshow(human)
        plt.show()
        catg[catg==255] = 0
        plt.imshow(catg)
        plt.show()

        print(image.shape, catg.shape, human.shape)
        # print(image.shape, catg.shape)

