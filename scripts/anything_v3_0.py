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
os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"
os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "1"

os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"

import click
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline as StableDiffusionPipeline
from pathlib import Path


@click.command()
@click.option("--token")
@click.option(
    "--prompt",
    default="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden",
)
@click.option("--repeat", default=32)
@click.option("--output", default="output")
def benchmark(token, prompt, repeat, output):
    pipe = StableDiffusionPipeline.from_pretrained(
        "Linaqruf/anything-v3.0",
        use_auth_token=token,
        revision="diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    output_dir = Path(output).joinpath("anything_v3_0")
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in range(repeat):
        images = pipe(prompt).images
        for i, image in enumerate(images):
            image.save(output_dir.joinpath(f"{r:03d}-{i:02d}.png"))


if __name__ == "__main__":
    benchmark()
