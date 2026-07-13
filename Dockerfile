FROM nvidia/cuda:12.3.2-cudnn9-devel-rockylinux9 AS compile-image

# install required command-line tools
RUN dnf install -y procps-ng

# install python 3.12 and git
RUN dnf install -y python3.12 && dnf clean all
RUN dnf install -y python3.12-pip && dnf clean all
RUN dnf install -y git && dnf clean all
RUN ln -s /usr/bin/python3.12 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.12 /usr/local/bin/python
RUN python3 --version

# install cuSPARSELt
RUN dnf install -y libcusparselt0 libcusparselt-devel && \
    ln -sf /usr/lib64/libcusparseLt.so.0 /usr/local/cuda/lib64/libcusparseLt.so.0

# Proseg
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN cargo install proseg
RUN ln -sf /usr/local/cargo/bin/proseg /usr/local/bin/proseg

# install spatialdata
RUN python3 -m pip install --no-cache-dir "spatialdata==0.8.0" "spatialdata-io>=0.6" "anndata>=0.10.9,<0.13.0" "pyarrow>=15.0.0"

# install MerXen
RUN python3 -m pip install --no-cache-dir git+https://github.com/bourdenxlab/MerXen@a2d95c1

CMD ["python3"]
