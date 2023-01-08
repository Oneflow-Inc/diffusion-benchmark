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
import PIL
import requests
from io import BytesIO
from pathlib import Path
import oneflow as torch
from diffusers import (
    OneFlowStableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline,
)

@click.command()
@click.option("--token")
@click.option(
    "--prompt", default="Face of a yellow cat, high resolution, sitting on a park bench"
)
@click.option("--image", default="stable_diffusion_inpainting_image.png")
@click.option("--mask", default="stable_diffusion_inpainting_mask_image.png")
@click.option("--repeat", default=32)
@click.option("--output", default="output")
def benchmark(token, prompt, image, mask, repeat, output):
    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    init_image = PIL.Image.open(image).resize((512, 512))
    mask_image = PIL.Image.open(mask).resize((512, 512))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        use_auth_token=token,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    Path(output).mkdir(parents=True, exist_ok=True)
    with torch.autocast("cuda"):
        for r in range(repeat):
            images = pipe(prompt, image=init_image, mask_image=mask_image).images
            for i, image in enumerate(images):
                image.save(f"{output}/stable_diffusion_inpainting-{r:03d}-{i:02d}.png")


if __name__ == "__main__":
    benchmark()
