ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Java is needed by the PTB tokenizer and METEOR used during captioning training.
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 git ninja-build default-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /samsung2023_iqa_solution
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fvcore==0.1.5.post20221221 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac' \
    && python -m spacy download en_core_web_sm
COPY . .
