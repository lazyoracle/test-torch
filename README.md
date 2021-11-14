# test-torch: Barebones repository to test installation of PyTorch with GPU functionality

## Requirements
* NVIDIA CUDA GPU enabled system
* Miniconda

## Usage

### Check out the code

```bash
git clone https://github.com/lazyoracle/test-torch
cd test-torch
```

### Build the environment

The `environment.yml` file works as is for Windows and Linux but **for MacOS, remove CUDA dependency using the following command**:

```bash
sed '/cuda/d' environment.yml | tee environment.yml
```

Creating a new `conda` environment is then straightforward using:

```bash
conda env create -f environment.yml
```

### Test your PyTorch installation

```bash
conda activate torch
pytest src/
```

If you do not have a supported GPU in your system:

```bash
pytest src/ -m "not needs_gpu"
```

If you just want to run the train and test script:

```bash
python src/gputorch.py -p <path> -e <epochs>
```