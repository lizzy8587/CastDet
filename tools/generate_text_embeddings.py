#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   generate_text_embeddings.py
@Version      :   1.0
@Time         :   2024/07/20 21:06:42
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   generate text embedding via CLIP text encoder
'''
import argparse
import open_clip
import torch
import numpy as np

VISDRONEZSD_CATEGORIES=('airplane', 'baseballfield', 'bridge', 'chimney', 'dam',
                        'Expressway-Service-area', 'Expressway-toll-station',
                        'golffield', 'harbor', 'overpass', 'ship', 'stadium',
                        'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                        'airport', 'basketballcourt', 'groundtrackfield', 'windmill')

RS_POOL = ['airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area',
           'Expressway-toll-station', 'golffield', 'harbor', 'overpass', 'ship', 'stadium',
           'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'airport', 'basketballcourt',
           'groundtrackfield', 'windmill', 'low scattered building', 'industrial area', 'desert',
           'circular farmland', 'building', 'steelsmelter', 'intersection', 'cloud',
           'red structured factory building', 'bare land', 'church', 'plasticgreenhouse',
           'dense tall building', 'grassland', 'thermal power station', 'refinery',
           'natural sparse forest land', 'beach', 'artificial dense forest land', 'vegetable plot',
           'natural dense forest land', 'mountain', 'wetland', 'river', 'sea ice',
           'blue structured factory building', 'chaparral', 'regular farmland',
           'medium density scattered building', 'artificial sparse forest land', 'water',
           'island', 'crossroads', 'terrace', 'forest', 'sparse residential', 'solar power station',
           'fish pond', 'commercial area', 'lrregular farmland', 'mobile home park', 'medium residential',
           'parking lot', 'roundabout', 'sewage plant-type-two', 'harbor', 'thermal power plant',
           'construction site', 'scattered blue roof factory building', 'snowberg', 'palace', 'lake',
           'scattered red roof factory building', 'rectangular farmland',
           'medium density structured building', 'sewage plant-type-one', 'sparse residential area',
           'square', 'dense residential']

def generate_clip_text_embeddings(save_path, model_path, text_queries, add_bg=False):
    """generate text embedding via CLIP text encoder"""

    text_queries = [f"a photo of {i}" for i in text_queries]
    model_name = 'RN50' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    ckpt = torch.load(model_path, map_location="cpu")
    message = model.load_state_dict(ckpt, strict=False)
    print(message)

    model = model.cuda().eval()
    text = tokenizer(text_queries)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.cuda())
        if add_bg:
            text_features = torch.cat([text_features, text_features.mean(dim=0, keepdim=True)], dim=0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    np.save(save_path, text_features.detach().cpu().float())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CLIP text embeddings')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the numpy file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the CLIP model checkpoint')
    parser.add_argument('--text_queries', nargs='+', default=VISDRONEZSD_CATEGORIES, help='List of text queries')
    parser.add_argument('--add_bg', action='store_true', help='Add background mean to text features')

    args = parser.parse_args()

    generate_clip_text_embeddings(
        save_path=args.save_path,
        model_path=args.model_path,
        text_queries=args.text_queries,
        add_bg=args.add_bg
    )