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
from diffusers import OneFlowAutoencoderKL as AutoencoderKL
from pathlib import Path
from diffusers.testing_oneflow_utils import floats_tensor
from tqdm import tqdm
import time


class VaePostProcess(torch.nn.Module):
    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


class VaeGraph(torch.nn.Graph):
    def __init__(self, vae_post_process) -> None:
        super().__init__()
        self.vae_post_process = vae_post_process

    def build(self, latents):
        return self.vae_post_process(latents)


@click.command()
@click.option("--token")
@click.option("--repeat", default=1000)
@click.option("--sync_interval", default=50)
def benchmark(token, repeat, sync_interval):
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            use_auth_token=token,
            revision="fp16",
            torch_dtype=torch.float16,
            subfolder="vae",
        )
        vae_post_process = VaePostProcess(vae)
        vae_post_process.to("cuda")
        vae_post_process.eval()
        vae_post_process_graph = VaeGraph(vae_post_process)
        batch_size = 1
        num_channels = 4
        sizes = (64, 64)
        noise = (
            floats_tensor((batch_size, num_channels) + sizes)
            .to("cuda")
            .to(torch.float16)
        )
        vae_post_process_graph(noise)
        torch._oneflow_internal.eager.Sync()

        t0 = time.time()
        for r in tqdm(range(repeat)):
            out = vae_post_process_graph(noise)
            if r == repeat - 1 or r % sync_interval == 0:
                torch._oneflow_internal.eager.Sync()
        t1 = time.time()
        duration = t1 - t0
        throughput = repeat / duration
        print(
            f"Finish {repeat} steps in {duration:.3f} seconds, average {throughput:.2f}it/s"
        )


if __name__ == "__main__":
    benchmark()
