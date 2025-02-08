import os.path as osp
from typing import List, Union
from mmdet.datasets import CocoDataset

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class NWPU45Dataset(CocoDataset):
    METAINFO = {
        'classes': ('golf_course', 'railway', 'rectangular_farmland', 'intersection',
                     'industrial_area', 'tennis_court', 'runway', 'commercial_area', 
                     'railway_station', 'ship', 'terrace', 'harbor', 'meadow', 'lake', 
                     'overpass', 'stadium', 'circular_farmland', 'airplane', 'airport', 'ground_track_field', 'forest', 'medium_residential', 'desert', 
                     'parking_lot', 'bridge', 'basketball_court', 'palace', 'storage_tank', 
                     'baseball_diamond', 'sparse_residential', 'church', 'mountain', 
                     'dense_residential', 'chaparral', 'mobile_home_park', 'island', 
                     'snowberg', 'sea_ice', 'roundabout', 'thermal_power_station', 'beach', 
                     'wetland', 'freeway', 'river', 'cloud'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
                    (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
                    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
                    (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
                    (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
                    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
                    (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
                    (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
                    (207, 138, 255), (151, 0, 95), (9, 80, 61)]
    }

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = img_info.get("caption", self.metainfo['classes'])
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info