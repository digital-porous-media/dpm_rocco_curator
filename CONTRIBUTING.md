# Contributing to Rocco

Thank you for your interest in contributing to Rocco! We welcome bug reports, feature requests, and pull requests from the community.

## Code of Conduct

We are committed to providing a welcoming and respectful environment for all contributors. Please treat all contributors with respect and dignity.

## Getting Started

### Prerequisites
- Python 3.9+
- `pip` and `git`

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dpm-rocco-curator.git
   cd dpm-rocco-curator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in editable mode with dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Copy and configure `.env`:**
   ```bash
   cp .env.example .env
   # Edit .env with your LLM provider credentials and configuration
   ```

## Making Changes

### Code Style

We use `black` and `isort` for code formatting. Before committing, run:

```bash
black . --line-length 100
isort .
```

Or combine them:
```bash
black . && isort .
```

### Testing

Run tests with pytest:

```bash
pytest tests/
```

For verbose output:
```bash
pytest -v tests/
```

### Running the UI Locally

To test your changes in the Streamlit app:

```bash
streamlit run rocco_ui.py
```

## Submitting Changes

### Creating a Branch

Create a branch for your feature or bug fix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-name
```

### Commit Messages

Write clear, descriptive commit messages:

```bash
git commit -m "Add feature: brief description of what was added"
git commit -m "Fix: description of what was fixed"
```

### Pushing and Creating a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a pull request on the main repository with:
   - A clear title and description
   - Reference to any related issues (e.g., "Closes #123")
   - Description of testing you've done

## Adding Support for a New LLM Provider

To add support for a new OpenAI-compatible LLM provider:

1. **Add the provider's base URL to `src/llm/client.py`:**
   ```python
   PROVIDER_URLS = {
       ...
       "newprovider": "https://api.newprovider.com/v1",
   }
   ```

2. **Update `.env.example`** with the new provider and its model names.

3. **Update the README** to document the new provider.

4. **Test** with a sample prompt using the new provider.

## Reporting Bugs

When reporting a bug, please include:

- Your Python version (`python --version`)
- Your operating system
- The exact steps to reproduce
- Expected and actual behavior
- Any error messages or logs

## Feature Requests

Before suggesting a feature, please:

- Check the existing issues and pull requests
- Describe the use case and expected behavior
- Consider whether it aligns with Rocco's core mission (dataset description curation)

## Questions or Discussion?

Feel free to open an issue for questions or discussions about the project.

---

Thank you for contributing to Rocco!
