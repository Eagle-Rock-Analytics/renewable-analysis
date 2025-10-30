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



## Development Workflow

### Working with `pre-commmit`

Please note that `pre-commit` is strict. It may be challenging to work with at first.

When you go to run `git commit -m "my message"` `pre-commmit` will be run BEFORE the commmit is run.

This means a good workflow will likely look like:

```bash
# add your changes
git add <my-file>

# attempt to commit them
git commmit -m "message"

# if pre-commit does not pass (almost never on the first try)
# check the changes that it made, or go fix the ones it couldn't fix.
git status

# add changes from pre-commmit
git add <my-file>

# try to commmit again
git commit -m "message"

# if you're successful, now you can push
git push
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific tests
pytest tests/test_example.py
pytest -m "not slow"  # Skip slow tests
```

### Code Quality

**Python files:**
```bash
# Format and lint (pre-commit does this automatically)
black src/ tests/
isort src/ tests/
ruff check src/ tests/

# Run all pre-commit hooks manually
pre-commit run --all-files
```

**Jupyter notebooks:**
```bash
nbqa black notebooks/
nbqa isort notebooks/
nbqa ruff notebooks/
```

### Security Scanning

```bash
# Scan for secrets
detect-secrets scan --baseline .secrets.baseline

# Vulnerability scanning
bandit -r src/
```

## Environment Management

### Adding Dependencies

1. **For runtime dependencies** (your package needs these to work):
   ```bash
   # Add to pyproject.toml [project] dependencies
   pip install package-name
   ```

   > [!TIP]
   > You **DO NOT** have to specify a minimum version for your package, in fact you should not unless there is good reason to do so (like breaking changes, security risks, etc.)
   >
   > After adding new dependencies, always re-run `pip install -e ".[dev,test]"` (or the appropriate install command) to ensure your environment is up to date.

2. **For development dependencies** (testing, linting, etc.):
   ```bash
   # Add to pyproject.toml [project.optional-dependencies.dev]
   pip install package-name
   ```

3. **Update environment file** (optional, for conda users):
   ```bash
   mamba env export > environment.yml
   ```

   > [!NOTE]
   > This is particularly useful if you're sharing exact environments across people, but for the most part pip and `pyproject.toml` will handle this.

### Useful Environment Commands

```bash
# Update existing environment
mamba env update -f environment.yml

# List all environments
mamba env list

# Remove environment (start fresh)
mamba env remove -n renewables-analysis
```
