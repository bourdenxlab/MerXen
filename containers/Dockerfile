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

# Install the pinned dependency tree from the lockfile, then the checked-out
# MerXen source without re-resolving its dependencies, so the image matches
# CI. uv is required: pip rejects the lock (ResolutionImpossible) because
# cell_type_mapper declares abc_atlas_access as an unpinned git URL while the
# lock pins a commit.
RUN python3 -m pip install --no-cache-dir "uv==0.11.7"
WORKDIR /opt/merxen
COPY requirements/requirements.lock ./requirements/requirements.lock
RUN uv pip install --system --python /usr/bin/python3.12 --no-cache -r requirements/requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --python /usr/bin/python3.12 --no-cache --no-deps .

CMD ["python3"]
