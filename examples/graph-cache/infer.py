import os

import argparse
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument("--cached", type=str, default="./oneflow_saved_pipe")
    parser.add_argument(
        "--load",
        default=False,
        action="store_true",
        help="If specified, load from cached",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="If specified, save to cached",
    )
    args = parser.parse_args()
    return args


args = parse_args()

model = "CompVis/stable-diffusion-v1-4"
if args.load:
    # Note: restore the cache by setting the pretrain path to a cache path
    model = args.cached
    print(f"will load pipe from: {args.cached}")
pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    model,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe = pipe.to("cuda")

output_dir = "oneflow-sd-output"
os.makedirs(output_dir, exist_ok=True)

# Note: enable graph-related cache
pipe.set_unet_graphs_cache_size(10)
pipe.enable_graph_share_mem()
pipe.enable_save_graph()


def do_infer(n):
    with torch.autocast("cuda"):
        for i in [2, 1, 0]:
            for j in [2, 1, 0]:
                width = 768 + 128 * i
                height = 768 + 128 * j
                prompt = args.prompt
                images = pipe(prompt, width=width, height=height).images
                for i, image in enumerate(images):
                    prompt = prompt.strip().replace("\n", " ")
                    dst = os.path.join(
                        output_dir, f"{prompt[:100]}-{n}-{width}-{height}.png"
                    )
                    image.save(dst)


for n in range(2):
    do_infer(n)
if args.save:
    # Note: graph cache will be saved with the weight
    print(f"start saving pipe to: {args.cached}")
    os.makedirs(args.cached, exist_ok=True)
    pipe.save_pretrained(args.cached)
    pipe.save_graph(args.cached)
