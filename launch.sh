HUGGINGFACE_CACHE_ROOT_DIR=${DIFFUSION_BENCHMARK_HF_HOME-`pwd`/huggingface_cache}
docker run -ti --rm --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --privileged -v $HUGGINGFACE_CACHE_ROOT_DIR:/root/.cache/huggingface -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} diffusion-benchmark${DIFFUSION_BENCHMARK_IMAGE_SUFFIX} /bin/bash

