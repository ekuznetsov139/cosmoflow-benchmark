# CosmoFlow TensorFlow Keras benchmark implementation

This is a an implementation of the
[CosmoFlow](https://arxiv.org/abs/1808.04728) 3D convolutional neural network
for benchmarking. It is written in TensorFlow with the Keras API and uses
[Horovod](https://github.com/horovod/horovod) for distributed training.

## Datasets

A dataset tarball is available via globus at:

https://app.globus.org/file-manager?origin_id=d0b1b73a-efd3-11e9-993f-0a8c187e8c12&origin_path=%2F

This is a 2.2 TB tar file containing 1027 `TFRecord` files, each representing a simulated universe with 64 sub-volume samples.

## Running the benchmark

Submission scripts are in `scripts`. YAML configuration files go in `configs`.

### Running at NERSC

`sbatch -N 64 scripts/train_cori.sh`
