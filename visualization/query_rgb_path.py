#!/usr/bin/env python3
"""
Query RGB image path from scene_image_sequences JSON.
Usage: python3 query_rgb_path.py --scene scene0089_00 --frame_idx 00005
"""
import pdb
import json
import argparse
import os


def query_rgb_path(scene_name, frame_idx, json_path):
    """Query RGB path from sequences dict. Returns path or empty string if not found."""
    if not os.path.exists(json_path):
        return ""

    with open(json_path, 'r') as f:
        sequences = json.load(f)

    if scene_name not in sequences:
        return ""

    # print(f"🔍 Found scene '{scene_name}' in sequences JSON, querying frame index '{frame_idx}'...")
    scene_sequences = sequences[scene_name]
    img_path = scene_sequences[0][int(frame_idx)]

    return img_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Scene name (e.g., scene0089_00)')
    parser.add_argument('--frame_idx', required=True, help='Frame index (e.g., 00005)')
    parser.add_argument('--json_path',
                        default='visualization/scene_image_sequences_all_30.json',
                        help='Path to JSON sequences file')
    args = parser.parse_args()

    result = query_rgb_path(args.scene, args.frame_idx, args.json_path)
    print(result)
