# test-torch: Barebones repository to test installation of PyTorch with GPU functionality

## Requirements
* NVIDIA CUDA GPU enabled system
* Miniconda

## Usage

To create a conda environment and test PyTorch GPU integration:

```bash
git clone https://github.com/lazyoracle/test-torch
cd test-torch
conda env create -f environment.yml
conda activate torch
pytest src/
```

If you just want to run the train and test script:

```bash
python src/gputorch.py -p <path> -e <epochs>
```