---
layout: default
title: "Installation"
---

## Stable release (easiest)

CellRegMap is implemented as a Python package.
To install CellRegMap using the pip python installer, enter:

    pip install cellregmap

in your command line.

## Developmental mode

To use the latest features of CellRegMap you can install the latest version from GitHub by entering:

    git clone https://github.com/limix/CellRegMap.git
    cd CellRegMap
    pip install -e .

in your command line.

## Installation using a Docker image 
If you use Docker, you can also pull the [pre-build image from dockerhub](https://hub.docker.com/r/annasecuomo/cellregmap).

<!-- you can build an image using the provided Dockerfile:

docker build -t mofa2 .
You will then be able to use R or Python from the container.

docker run -ti --rm -v $DATA_DIRECTORY:/data mofa2 R
#                   ^
#                   |
#                    use `-v` to map a folder on your machine to a container directory
The command above will launch R with MOFA2 and its dependencies installed while mounting $DATA_DIRECTORY to the container. --> 

<!-- ## Running tests

From your command line, enter

    python setup.py test --> 
