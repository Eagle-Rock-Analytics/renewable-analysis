# Renewables Analysis

A collection of tools for performing analysis on the data generated in [this repository](https://github.com/Eagle-Rock-Analytics/renewable-profiles)



## Set Up Your Development Environment

**Quick setup (recommended):**
```bash
make setup  # Configure and install environment
mamba activate renewables-analysis
make all    # Set up pre-commit tools
```

For more shortcuts, run `make` or `make help`.

**Manual setup:**

#### Option A: Using Mamba (Recommended)

Mamba is faster and more reliable than conda:

```bash
# Install mamba if you don't have it
conda install mamba -n base -c conda-forge

# Create and activate environment
mamba env create -f environment.yml
mamba activate renewables-analysis

# Install project in development mode
pip install -e ".[dev,test]"
```

#### Option B: Using Conda

```bash
conda env create -f environment.yml
conda activate renewables-analysis
pip install -e ".[dev,test]"
```

#### Option C: Using pip + virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,test]"
```

### Add climakitae

After environment is created, add tools for loading climate data catalog from [climakitae](https://github.com/cal-adapt/climakitae/tree/main)

With the environment active, run:
```
pip install https://github.com/cal-adapt/climakitae/archive/refs/tags/1.4.0.zip
```

### 4. Initialize Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files  # Optional: run on all existing files
```
