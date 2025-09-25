# Template Basic

A comprehensive Python project template for Eagle Rock Analytics with modern development practices, code quality tools, and CI/CD workflows.

## Features

- 🐍 **Python 3.12+** support
- 📦 **Modern packaging** with `pyproject.toml`
- 🔧 **Pre-commit hooks** for code quality and security
- 🚀 **GitHub Actions** CI/CD workflows
- 🧪 **Testing** with pytest and coverage reporting
- 📊 **Code coverage** with codecov integration
- 🔒 **Security** scanning for API keys and vulnerabilities
- 📝 **Code formatting** with Black, Ruff, and isort
- 📋 **Issue and PR templates**
- 📚 **Structured project layout**

## Project Structure

```
template-basic/
├── configure_project.sh        # Project configuration tool
├── environment.yml             # Conda/mamba environment definition
├── .github/                    # GitHub configuration files
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   ├── PULL_REQUEST_TEMPLATE   # PR template
│   └── workflows/              # CI/CD workflows
├── .gitignore                  # Git ignore patterns
├── Makefile                    # Development shortcuts
├── notebooks/                  # Jupyter notebooks
│   └── example.ipynb           # Example notebook
├── .pre-commit-config.yaml     # Pre-commit configuration
├── pyproject.toml              # Project definition and dependencies
├── README.md                   # This file
├── src/                        # Python source code
│   ├── example.py              # Example module
│   └── __init__.py             # Package initialization
└── tests/                      # Test suite
    ├── conftest.py             # Pytest configuration
    └── test_example.py         # Example tests
```

## Quick Start

### 1. Create Your Project from This Template

1. Click "Use this template" on GitHub
2. Name your new repository
3. Clone your new repository:
   ```bash
   git clone https://github.com/Eagle-Rock-Analytics/your-new-project.git
   cd your-new-project
   ```

### 2. Configure Your Project Name (Recommended)

Use the provided script to update all project references to your new project name:

```bash
# Basic usage
./configure_project.sh my-awesome-project

# With backup (recommended)
./configure_project.sh --backup my-awesome-project

# Preview changes without modifying files
./configure_project.sh --dry-run my-awesome-project

# Restore from backups if needed
./configure_project.sh --restore
```

**What this script does:**
- Updates all references from `template-basic`/`template_basic` to your project name
- Renames source directories to match
- Updates imports in code and tests
- Ensures consistency across all configuration files

### 3. Set Up Your Development Environment

**Quick setup (recommended):**
```bash
make setup  # Configure and install environment
mamba activate template-basic
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
mamba activate template-basic

# Install project in development mode
pip install -e ".[dev,test]"
```

#### Option B: Using Conda

```bash
conda env create -f environment.yml
conda activate template-basic
pip install -e ".[dev,test]"
```

#### Option C: Using pip + virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,test]"
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
mamba env remove -n template-basic
```

## CI/CD & GitHub Integration

### Included GitHub Actions

- **Code Quality**: Linting, formatting, and type checking
- **Testing**: Multi-version Python testing
- **Security**: Vulnerability and secret scanning
- **Coverage**: Automated coverage reporting

### Setting up Codecov (Optional)

1. Sign up at [codecov.io](https://codecov.io) with your GitHub account
2. Add your repository
3. Add `CODECOV_TOKEN` to repository secrets:
   - Repository Settings → Secrets and variables → Actions
   - New secret: CODECOV_TOKEN
4. Add to your `.github/workflows/ci.yml`:
   ```yaml
   - name: Upload coverage to Codecov
     uses: codecov/codecov-action@v4
     with:
       files: ./coverage.xml
       fail_ci_if_error: true
       token: ${{ secrets.CODECOV_TOKEN }}
   ```

### Branch Protection

Recommended settings for `main` branch:
1. Repository Settings → Branches → Add rule
2. Enable:
   - ✅ Require pull request reviews
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date
   - ✅ Include administrators

## Customization

### Essential Updates

1. **`pyproject.toml`**: Update `description`, `authors`, and `urls`
2. **This README**: Replace with your project-specific documentation
3. **Dependencies**: Add your actual runtime dependencies

### Project Structure

- **`src/`**: Your Python modules and packages
- **`tests/`**: Unit and integration tests
- **`notebooks/`**: Exploratory analysis and documentation
- **`.github/`**: CI/CD workflows and templates

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run quality checks: `pre-commit run --all-files`
5. Commit and push: `git commit -m "Add feature" && git push origin feature-name`
6. Create a Pull Request

## Troubleshooting

### Common Issues

**Pre-commit hooks failing:**
```bash
pre-commit autoupdate
pre-commit clean
pre-commit install
```

**Environment conflicts:**
```bash
mamba env remove -n template-basic
mamba env create -f environment.yml
```

**Import errors in notebooks/tests:**
```bash
pip install -e .
```

**Slow conda/mamba operations:**
```bash
# Switch to mamba for faster dependency resolution
conda install mamba -n base -c conda-forge
```

## Why This Template?

This template follows modern Python best practices:

- **`pyproject.toml`**: Standard for Python packaging (PEP 518)
- **Pre-commit hooks**: Catch issues before they reach CI
- **Ruff**: Fast, modern linting (replaces flake8, pylint, etc.)
- **Pytest**: Industry standard testing framework
- **GitHub Actions**: Integrated CI/CD
- **Security scanning**: Automated vulnerability detection

## Support

- 📋 **Issues**: Check existing issues or create a new one
- 📧 **Contact**: Reach out to the Eagle Rock Analytics development team
- 📚 **Documentation**: See individual tool documentation for detailed usage

---

**Happy coding! 🚀**
