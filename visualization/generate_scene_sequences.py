import pdb
import os
import pickle
import json
import argparse


def _map_to_local_img_path(scene_name, img_path, local_posed_roots):
    if os.path.exists(img_path):
        return img_path

    normalized = str(img_path).replace('\\', '/')
    basename = os.path.basename(normalized)

    candidates = []
    for root in local_posed_roots:
        candidates.append(os.path.join(root, scene_name, basename))

    marker = '/posed_images/'
    if marker in normalized:
        tail = normalized.split(marker, 1)[1]
        for root in local_posed_roots:
            candidates.append(os.path.join(root, tail))

    marker_typo = '/posed_imagse/'
    if marker_typo in normalized:
        tail = normalized.split(marker_typo, 1)[1]
        for root in local_posed_roots:
            candidates.append(os.path.join(root, tail))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0] if len(candidates) > 0 else img_path


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst) // chunk_size * chunk_size, chunk_size)]


def build_sequences(occscannet_root, phase, data_tag, num_frames, local_posed_roots):
    if data_tag == 'base':
        subscenes_list = os.path.join(occscannet_root, f'{phase}_online.txt')
    else:
        subscenes_list = os.path.join(occscannet_root, f'{phase}_mini_online.txt')

    if not os.path.exists(subscenes_list):
        raise FileNotFoundError(f"Scenes list not found: {subscenes_list}")

    with open(subscenes_list, 'r') as f:
        used_subscenes = [l.strip() for l in f.readlines() if l.strip()]

    out = {}
    for name in used_subscenes:
        scene_pkg_pth = os.path.join(occscannet_root, 'global_occ_package', f'{name}.pkl')
        if not os.path.exists(scene_pkg_pth):
            print(f'Warning: scene pkg not found: {scene_pkg_pth} — skipping')
            continue

        with open(scene_pkg_pth, 'rb') as f:
            scene_pkg = pickle.load(f)

        valid_img_paths = scene_pkg.get('valid_img_paths', [])
        # pdb.set_trace()
        mapped = [_map_to_local_img_path(name, p, local_posed_roots) for p in valid_img_paths]
        pdb.set_trace()
        # # sort by frame id extracted from filename
        # try:
        #     sorted_image_paths = sorted(mapped, key=lambda x: int(os.path.basename(x).split('.')[0]))
        # except Exception:
        #     sorted_image_paths = mapped
        sorted_image_paths = mapped

        sequences = chunk_list(sorted_image_paths, num_frames)
        out[name] = sequences

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all', help='Phase to generate: train, test, val, or all (default: all)')
    parser.add_argument('--data_tag', type=str, default='base')
    parser.add_argument('--num_frames', type=int, default=30)
    parser.add_argument('--occscannet_root', type=str, default='data/scene_occ')
    parser.add_argument('--out_dir', type=str, default='visualization')
    args = parser.parse_args()

    local_posed_roots = [
        'data/occscannet/posed_images',
        'data/occscannet/posed_imagse',
    ]

    # Determine which phases to process
    if args.phase == 'all':
        phases = ['train', 'test', 'val']
    else:
        phases = [args.phase]

    # Build sequences for each phase and merge
    all_sequences = {}
    for phase in phases:
        try:
            sequences = build_sequences(
                occscannet_root=args.occscannet_root,
                phase=phase,
                data_tag=args.data_tag,
                num_frames=args.num_frames,
                local_posed_roots=local_posed_roots,
            )
            all_sequences.update(sequences)
            print(f'Loaded {len(sequences)} scenes from phase: {phase}')
        except FileNotFoundError as e:
            print(f'Phase {phase} not found: {e}')

    os.makedirs(args.out_dir, exist_ok=True)

    # Use 'all' in filename if processing all phases
    file_tag = 'all' if args.phase == 'all' else args.phase
    pkl_path = os.path.join(args.out_dir, f'scene_image_sequences_{file_tag}_{args.num_frames}.pkl')
    json_path = os.path.join(args.out_dir, f'scene_image_sequences_{file_tag}_{args.num_frames}.json')

    with open(pkl_path, 'wb') as f:
        pickle.dump(all_sequences, f)

    # JSON-friendly: convert to lists
    sequences_json = {k: v for k, v in all_sequences.items()}
    with open(json_path, 'w') as f:
        json.dump(sequences_json, f, indent=2)

    total_scenes = len(all_sequences)
    total_seqs = sum(len(v) for v in all_sequences.values())
    print(f'\nWrote {pkl_path} and {json_path}')
    print(f'Total scenes: {total_scenes}, total sequences: {total_seqs}')


if __name__ == '__main__':
    main()
