# Parameter-Efficient Fine-Tuning with LoRA

[my-website]: https://AJG91.github.io "my-website"

This repository demonstrates how to efficiently fine tune a model using Low-rank adaptation (LoRA).

## Getting Started

* This project relies on `python=3.12`. It was not tested with different versions
* Clone the repository to your local machine
* Once you have, `cd` into this repo and create the virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate lora-fine-tune-env`
* Install the packages in the repo root directory using `pip install -e .` (you only need the `-e` option if you intend to edit the source code in `lora_fine_tune/`)

## Example

See [my website][my-website] for examples on how to use this code.

## Citation

If you use this project, please use the citation information provided by GitHub via the **“Cite this repository”** button or cite it as follows:

```bibtex
@software{Garcia2026LoRAFineTune,
  author = {Alberto J. Garcia},
  title = {Parameter-Efficient Fine-Tuning with LoRA},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/AJG91/lora-fine-tune},
  license = {MIT}
}
```