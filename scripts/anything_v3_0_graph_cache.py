import os

os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"

os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"

import click
import oneflow as torch
from diffusers import (
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
    OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
)

from pathlib import Path

DEFAULT_PROMPT = "masterpiece, best quality, illustration, beautiful detailed, finely detailed, dramatic light, intricate details, 1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden"
DEFAULT_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"


@click.command()
@click.option("--token")
@click.option("--prompt", default=DEFAULT_PROMPT)
@click.option("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
@click.option("--guidance_scale", default=12)
@click.option("--repeat", default=8)
@click.option("--output", default="output")
@click.option("--min_height", default=512)
@click.option("--min_width", default=512)
@click.option("--max_height", default=768)
@click.option("--max_width", default=768)
@click.option("--size_step", default=128)
@click.option("--num_inference_steps", default=50)
def benchmark(
    token,
    prompt,
    negative_prompt,
    guidance_scale,
    repeat,
    output,
    min_height,
    min_width,
    max_height,
    max_width,
    size_step,
    num_inference_steps,
):
    model_id = "Linaqruf/anything-v3.0"
    sizes = [
        (height, width)
        for height in range(min_height, max_height + 1, size_step)
        for width in range(min_width, max_width + 1, size_step)
    ]

    sizes.reverse()

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_graph_compile_cache_size(len(sizes))
    pipe.enable_graph_share_mem()
    pipe = pipe.to("cuda")

    output_dir = Path(output).joinpath("anything_v3_0_graph_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in range(repeat):
        for height, width in sizes:
            images = pipe(
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images
            for i, image in enumerate(images):
                image.save(output_dir.joinpath(f"{width}x{height}-{r:03d}-{i:02d}.png"))


if __name__ == "__main__":
    benchmark()
