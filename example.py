"""Enhances the 3DGS renderings of one scene with the 3DGS-Enhancer checkpoint.

    python example.py --model_path checkpoints/3DGS-Enhancer --lq_root <generator output> \
        --gt_root <DL3DV scenes> --scene <scene name> --num_views 3 --output_path outputs/
"""
import argparse
import os
import sys
import tempfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules", "SVDFor3D"))

import torch
from PIL import Image

from inference_svd import DummyInferenceDataset
from pipelines.pipeline_3dgs_enhancer import Enhancer3DGSPipeline

WIDTH, HEIGHT, NUM_FRAMES = 960, 512, 26


def load_scene(lq_root, gt_root, scene, num_views):
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as scene_list:
        scene_list.write(scene + "\n")
        scene_list.flush()
        return DummyInferenceDataset(
            base_folders=lq_root,
            ref_folders=gt_root,
            width=WIDTH,
            height=540,
            sample_frames=NUM_FRAMES,
            scene_txt=scene_list.name,
            view_num=str(num_views),
        )


def load_window(sample, device):
    """Returns the 3DGS renderings with clean input views at both ends, the camera poses and the frame names."""
    lq = (sample["condition_pixel_values"] + 1.0) / 2.0
    gt = (sample["pixel_values"] + 1.0) / 2.0
    lq[0], lq[-1] = gt[0], gt[-1]
    top = (lq.shape[-2] - HEIGHT) // 2
    lq = lq[..., top:top + HEIGHT, :]
    names = [os.path.splitext(os.path.basename(path))[0] for path in sample["image_paths"]]
    return lq.to(device), sample["condition_T"].to(device), names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lq_root", type=str, required=True, help="dataset generator output")
    parser.add_argument("--gt_root", type=str, required=True, help="scenes with the input view images in images_4/")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda")
    pipeline = Enhancer3DGSPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device)
    pipeline.set_progress_bar_config(disable=True)
    dataset = load_scene(args.lq_root, args.gt_root, args.scene, args.num_views)

    out_dir = os.path.join(args.output_path, args.scene)
    os.makedirs(out_dir, exist_ok=True)
    # each window holds the renderings between two neighbouring input views
    windows = list(dict.fromkeys(tuple(sample) for sample in dataset.samples))
    written = set()
    for i, window in enumerate(windows):
        lq, poses, names = load_window(dataset[dataset.samples.index(list(window))], device)
        generator = torch.Generator(device).manual_seed(args.seed + i)
        frames = pipeline(lq, poses, generator=generator).frames
        frames = (frames * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for frame, name in zip(frames, names):
            if "_ref" not in name and name not in written:
                written.add(name)
                Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))
        print(f"window {i + 1}/{len(windows)} done", flush=True)
    print(f"enhanced frames saved to {out_dir}")
