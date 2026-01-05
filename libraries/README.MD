### Libraries

Here we store external libraries that we use in our project. We might make changes to them, so we custom install it via `pip install -e .` from the root directory.

---

## VeRL Data Preparation

Prepare forecasting datasets for VeRL training (output: parquet format).
First go to `cd verl/examples/data_preprocess/forecasting`.

### Load OpenForesight from HuggingFace

```bash
cd verl/examples/data_preprocess/forecasting

# Full dataset
python load_foresight.py --split train --output_dir data/

# Subsample
python load_foresight.py --split train --subsample 1000 --output_dir data/
```

**Output:** Parquet files ready for VeRL training in `data/` directory.

**For Training**:

Go to the following folder and check the readme there. 
```bash
cd libraries/verl/scripts/ours/testrun/
cat README.md
```