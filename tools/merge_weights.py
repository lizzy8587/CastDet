#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   merge_weights.py
@Version      :   1.0
@Time         :   2024/04/20 00:00:16
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''

import argparse
import torch

def merge_weights(clip_path, base_path, save_path, base_model='soft-teacher'):
    clip = torch.load(clip_path)#.state_dict()
    base = torch.load(base_path)
    save_dict = {}
    for k, v in base['state_dict'].items():
        if base_model == 'soft-teacher':
            save_dict[k] = v
        elif base_model == 'faster-rcnn':
            save_dict['teacher.' + k] = v
            save_dict['student.' + k] = v
    
    for k, v in clip.items():
        if k.startswith('visual.'):
            save_dict[k] = v
            
    print(save_dict.keys())
    torch.save(save_dict, save_path)

def main():
    parser = argparse.ArgumentParser(description="Merge weights from CLIP and a base detection model")
    parser.add_argument("--clip_path", type=str, required=True, help="Path to the CLIP model checkpoint")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the base model checkpoint")
    parser.add_argument("--save_path", type=str, required=True, help="Path where the merged model will be saved")
    parser.add_argument("--base_model", type=str, default="soft-teacher", choices=["soft-teacher", "faster-rcnn"], help="Base model type: 'soft-teacher' or 'faster-rcnn'")
    
    args = parser.parse_args()
    merge_weights(args.clip_path, args.base_path, args.save_path, args.base_model)

if __name__ == "__main__":
    main()
