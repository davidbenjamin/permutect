# Using this image will ensure a specific Python version (3.10) and specific CUDA Version (12.1)
# Note that `nvidiaDriverVersion` must be at least 525.60.13 on linux to support Cuda 12.1:
# https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
# Google cloud makes driver version recommendations for GCE VMs here:
# https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Overrides the default pytorch image workdir
WORKDIR /

COPY requirements.txt /
COPY setup.py /
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

ADD permutect/ /permutect

RUN pip install build
RUN python3 -m build --sdist
RUN pip install dist/*.tar.gz

CMD ["/bin/sh"]
