# Dependencies

It is recommended installing a fresh Conda environment and run python 3.8+.

```bash
conda create --name <my_env> python==3.8
conda activate <my_env>
```

Once activated, we need to install dependencies:

```bash
git clone https://github.com/Dpm-a/NLP-Large-Movies-Sentiment
cd src/
source requirements.sh
```

# Usage

In `Notebook` there will be a `.ipynb` file use for different trials on input data.

The only file to run is instead:

```bash
 cd src/
 python inference.py
```

## Optional Parameters

The script supports several optional parameters to customize the behavior. These parameters can be added when running the script from the command line. Below is the list of optional parameters:

- `-s` or `--sentence`: If a sentence is provided, a Transformer model will handle it and return a sentiment class.

- `-x` or `--x_test`: explicit set the column to accept as input data, default = `review`.

- `-y` or `--y_test`: explicit set the column to accept as target variable, default = `rating`.

- `-t` or `--transformers`: Inference on a complete dataset with Transformers(BERT).

### Usage Example

To run the script with optional parameters, use the following format:

```bash
cd src/
python inference.py -s "<your-text-here>"
```
