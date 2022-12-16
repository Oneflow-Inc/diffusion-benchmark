import os

os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

os.environ["ONEFLOW_KERENL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
os.environ["ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
# os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"
os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "1"

os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"

import click
import oneflow as torch
from diffusers import (
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
    OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
)

from pathlib import Path


@click.command()
@click.option("--token", help="auth token")
@click.option("--repeat", default=32, help="")
@click.option("--output", default="output", help="")
@click.option("--height", default=768, help="")
@click.option("--width", default=768, help="")
def benchmark(token, repeat, output, height, width):
    model_id = "stabilityai/stable-diffusion-2-1"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    Path(output).mkdir(parents=True, exist_ok=True)
    prompt = "a photo of an astronaut riding a horse on mars"
    for r in range(repeat):
        images = pipe(prompt, height=height, width=width).images
        for i, image in enumerate(images):
            image.save(f"{output}/stable_diffusion_2_1-{r:03d}-{i:02d}.png")


if __name__ == "__main__":
    benchmark()
