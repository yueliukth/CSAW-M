# used these as examples:
# https://github.com/kaust-vislab/python-data-science-project/blob/master/docker/Dockerfile
# https://hub.docker.com/r/anibali/pytorch/dockerfile
# https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile

# use this as base image
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ARG UID
ARG GID
ARG USER
ARG GROUP
ARG WANDB_KEY


# Make RUN commands use `bash --login`
SHELL [ "/bin/bash", "--login", "-c" ]

# install utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        tree \
        zip \
        git \
        curl \
        ca-certificates \
        sudo \
        bzip2 \
        libx11-6 \
        git \
        wget \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# install cron
RUN apt-get update && apt-get -y install cron


# Create a non-root user
# change your username AND
# change your uid (run id -u to learn it) 
# and gid (run id -g to learn it)
ENV USER $USER
ENV UID $UID
ENV GID $GID
ENV HOME /home/$USER

RUN addgroup --gid $GID $USER
RUN adduser --disabled-password \
   --gecos "Non-root user" \
   --uid $UID \
   --gid $GID \
   --home $HOME \
   $USER
 
# install miniconda
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3
# note that the version should exist in the repo https://repo.anaconda.com/miniconda/
ENV MINICONDA_VERSION latest
ENV CONDA_DIR=/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
   chmod +x ~/miniconda.sh && \
   ~/miniconda.sh -b -p $CONDA_DIR && \
   rm ~/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interactive shells
RUN conda init bash

# copy needed files
COPY env_linux_v2_new.yml $HOME

# build the conda environment using the yaml file
RUN conda update --name base --channel defaults conda && \
    conda env create -f $HOME/env_linux_v2_new.yml && \
    conda clean --all --yes

# activate the env
# RUN conda activate detect

# setting this env var helps with autocomplete
ENV SHELL=/bin/bash

# text file environment to be used in crontab
RUN env > /root/env.txt

WORKDIR $HOME/CSAW-M

# RUN wandb login
ENV WANDB_API_KEY=$WANDB_KEY

# add aliases
RUN echo 'alias gpu="watch -n 0.2 nvidia-smi"' >> ~/.bashrc \
    && echo 'refresh_bashrc="source ~/.bashrc"' >> ~/.bashrc \
    && echo 'alias home="cd $HOME"' >> ~/.bashrc \
    && echo 'alias csaw-m="cd $HOME/CSAW-M/"' >> ~/.bashrc \
&& source ~/.bashrc


# Run the command on container startup
CMD ["/bin/bash"]

USER root
