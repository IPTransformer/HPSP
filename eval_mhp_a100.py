import argparse
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
# from instance_data import COCOInstanceTestDataset
from utils.evaluation.parsing_eval_mhp import ParsingEvaluator, parsing_nms_base, parsing_nms_base_mm
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from tqdm import tqdm

from models import build_cihp_model


class COCOInstanceTestDataset(data.Dataset):
    def __init__(self, ann_file, root, image_thresh=0.0, ann_types='parsing', transforms=None, extra_fields={}):
        self.root = root
        self.coco = COCO(ann_file)
        # self.coco_dt = self.coco.loadRes(bbox_file) if bbox_file else None
        self.coco_dt = None
        self.image_thresh = image_thresh
        self.ann_types = ann_types
        self.transforms = transforms
        self.ids = sorted(self.coco.imgs.keys())  #[:10]
        self.extra_fields = extra_fields

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        category_ids = sorted(self.coco.getCatIds())
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(category_ids)}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        category_ids = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = category_ids

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # if self.coco_dt:
        #     ann_ids = self.coco_dt.getAnnIds(imgIds=img_id)
        #     anno = self.coco_dt.loadAnns(ann_ids)
        #     anno = [obj for obj in anno if obj['score'] > self.image_thresh]
        # else:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # anno = self.coco.loadAnns(ann_ids)

        try:
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
        except:
            file_name = self.coco.loadImgs(img_id)[0]['coco_url'].split('.org/')[-1]
        # img = np.asarray(Image.open(os.path.join(self.root, file_name)).convert('RGB'))
        _img = cv2.imread(os.path.join(self.root, file_name))
        bgr = _img[:, :, :3]
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


        return img, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        img_id = self.id_to_img_map[idx]
        img_data = self.coco.imgs[img_id]
        return img_data

    def pull_image(self, idx):
        """Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            idx (int): index of img to show
        Return:
            img
        """
        img_id = self.id_to_img_map[idx]

        try:
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
        except:
            file_name = self.coco.loadImgs(img_id)[0]['coco_url'].split('.org/')[-1]

        return Image.open(os.path.join(self.root, file_name)).convert('RGB')


def process_data(img, device, size=448):
    # print(temp_size)
    h, w = img.shape[:2]
    h_resize = size
    w_resize = size

    rescale_ratio = min(float(w_resize) / w, float(h_resize) / h)

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    # mask = cv2.resize(lbl, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    # pad
    h_pad = max((h_resize - h_new), 0)
    w_pad = max((w_resize - w_new), 0)
    p1 = h_pad // 2
    p2 = h_pad - p1
    p3 = w_pad // 2
    p4 = w_pad - p3

    img = np.pad(img, ((p1, p2), (p3, p4), (0, 0)), 'constant', constant_values=(128, 128))
    # mask = np.pad(mask, ((p1, p2), (p3, p4)), 'constant', constant_values=(255, 255))

    # mask = change11to9(mask)

    # normalize
    img = np.array(img).astype(np.float32)
    img /= 255.0
    img -= (0.485, 0.456, 0.406)
    img /= (0.229, 0.224, 0.225)

    # totensor
    img = np.array(img).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img).float()

    img = img.unsqueeze(0)

    image = img.to(device)

    return image, p1,p3,h_new,w_new


if __name__ == '__main__':
    json_file = 'datas/MHP_val.json'
    root = 'datas/LV-MHP-v2/val/images/'
    seg_root = 'datas/LV-MHP-v2/val/annotatons/'


    extra_fields = {}
    extra_fields['seg_root'] = seg_root
    extra_fields['semseg_format'] = 'mask'

    dataset = COCOInstanceTestDataset(json_file, root, extra_fields=extra_fields)
    print('dataset created!')



    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument('--backbone', type=str, default='Swin_S',
                        help='backbone name (default: resnet)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512, 640, 768, 960],
                        help='img size to bre trained')
    parser.add_argument('--nclass', type=int, default=59,
                        help='img size to bre trained')
    parser.add_argument('--ins', nargs='+', type=int,  default=[640],
                        help='img size to bre trained')
    # checking point
    parser.add_argument('--mix-ins', action='store_true', default=False,
                        help='whether use_checkpoint (default: False)')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')


    args = parser.parse_args()
    print(args)



    TP_THRESHOLD = 0.5
    RATIO_THRESHOLD = 0.00038
    NMS_THRESHOLD = 0.6



    device = torch.device('cuda:{}'.format(0))

    backbone = args.backbone
    build_model = build_cihp_model(backbone)
    model = build_model(backbone=backbone, num_classes=59, num_instances=18,
                        use_checkpoint=False)

    checkpoint = torch.load(args.checkname)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = model.to(device)
    model.eval()

    pars_results = []
    mask_results = {}
    APr_filename = {}
    APh_filename = {}

    ms = args.img_size

    output_pth = {}
    for i in tqdm(range(len(dataset))):
    # for i in range(10):

        # if i%100 ==0:
        #     print(i)
        parsing_tmp = []

        img = dataset[i][0]
        img_info = dataset.coco.loadImgs(i+1)

        img_id = img_info[0]['id']
        filename = img_info[0]['file_name'][:-4]

        h, w = img.shape[:2]
        area = h*w

        output_ins_mix = torch.zeros((1, 18, h, w), device=device)
        output_sem_mix = torch.zeros((1, 59, h, w), device=device)
        for size in ms:
            image, p1,p3,h_new,w_new = process_data(img, size=size, device=device)

            with torch.no_grad():
                output_ins, output_sem = model(image)

            output_ins = output_ins[:, :, p1:p1 + h_new, p3:p3 + w_new]
            output_sem = output_sem[:, :, p1:p1 + h_new, p3:p3 + w_new]
            output_ins = torch.nn.functional.interpolate(output_ins, size=(h, w), mode='bilinear', align_corners=True)
            output_sem = torch.nn.functional.interpolate(output_sem, size=(h, w), mode='bilinear', align_corners=True)

            output_sem_mix += output_sem
            if args.mix_ins:
                output_ins_mix += output_ins
            else:
                if size in args.ins:
                    output_ins_mix += output_ins


        # if args.mix_ins:
        output_ins_mix = output_ins_mix / len(args.ins)
        # output_sem_mix = output_sem_mix / len(ms)
        output_sem_mix = output_sem_mix / len(ms)
        pred_mask = output_sem_mix.max(1)[1]

        parsings, instance_scores, part_scores, human_ins = parsing_nms_base(pred_ins=output_ins_mix[0],
                                                                      pred_mask_brforeargmax=output_sem_mix,
                                                                    num_parsing=59, num_instance=18,
                                                                      nms_thresh=NMS_THRESHOLD,
                                                                      TP_THRESHOLD=TP_THRESHOLD)




        for ins_ in range(parsings.shape[0]):
            parsing = parsings[ins_]

            pars_results.extend(
                [
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "parsing": csr_matrix(parsing),
                        "score": instance_scores[ins_],
                    }
                ])
            # parsing_tmp.append(parsing[np.newaxis,:,:])
        # if len(parsing_tmp)>0:
        mask_results[img_id] = pred_mask[0].cpu().numpy()


    print('Finished forward computation. offical nms')

    pet_eval = ParsingEvaluator(
        parsingGt=dataset, parsingPred=pars_results, maskPred=mask_results,
        gt_dir=root, pred_dir=root, metrics=['APp', 'mIoU']
    )

    pet_eval.evaluate()
    pet_eval.accumulate()
    pet_eval.summarize()
















