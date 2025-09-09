# Contributing to Real-Time Fraud Detection System

Thank you for your interest in contributing to this fraud detection project! This document provides guidelines for contributing to the codebase.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/nigavictor/real_time_fraud_detection.git
   cd real_time_fraud_detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv fraud_detection_env
   source fraud_detection_env/bin/activate  # Linux/Mac
   # or
   fraud_detection_env\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run the test suite**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Run the demo pipeline**
   ```bash
   python demo_pipeline.py
   ```

## 📋 Contribution Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes** - Fix identified issues in the codebase
2. **Feature Enhancements** - Improve existing functionality
3. **New Features** - Add new capabilities to the system
4. **Documentation** - Improve code documentation and guides
5. **Performance Optimization** - Enhance system performance
6. **Test Coverage** - Add or improve unit tests

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Follow the existing code style
   - Add appropriate comments and docstrings
   - Write or update tests as needed

3. **Run quality checks**
   ```bash
   # Format code
   black .
   isort .
   
   # Run linting
   flake8 .
   
   # Run tests
   python -m pytest tests/ -v --cov=src
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new fraud detection feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues
   - Include screenshots for UI changes
   - Ensure CI passes

### Code Style Guidelines

- **Python Style**: Follow PEP 8
- **Imports**: Use isort for import organization
- **Formatting**: Use Black for code formatting
- **Line Length**: Maximum 88 characters
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type hints where appropriate

### Example Code Style

```python
def train_fraud_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    model_type: str = "xgboost"
) -> Dict[str, Any]:
    """
    Train a fraud detection model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        model_type: Type of model to train
        
    Returns:
        Dictionary containing trained model and metadata
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Implementation here
    pass
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete pipeline functionality

### Writing Tests

```python
import pytest
import pandas as pd
from src.modeling.model_trainer import FraudModelTrainer

class TestFraudModelTrainer:
    """Test suite for FraudModelTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = FraudModelTrainer()
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        self.sample_labels = pd.Series([0, 1, 0, 1, 0])
    
    def test_model_training(self):
        """Test that model training works correctly."""
        result = self.trainer.train_model(
            self.sample_data, 
            self.sample_labels, 
            'logistic_regression'
        )
        
        assert result is not None
        assert 'model' in result
        assert 'metadata' in result
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model_trainer.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run performance tests
python -m pytest tests/ -m performance
```

## 📚 Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use type hints for function parameters and return values
- Include examples in docstrings where helpful
- Document any complex algorithms or business logic

### README Updates

If your contribution affects the user interface:

- Update installation instructions if needed
- Add new usage examples
- Update the feature list
- Modify performance benchmarks if applicable

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the bug
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, package versions
6. **Error Messages**: Full error messages and stack traces

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.11.0
- OS: Ubuntu 22.04
- Package versions: (run `pip freeze`)

## Error Messages
```
Full error message here
```
```

## 💡 Feature Requests

For new features, please provide:

1. **Use Case**: Why is this feature needed?
2. **Description**: Detailed description of the feature
3. **Proposed Implementation**: How should it work?
4. **Alternatives**: Any alternative solutions considered
5. **Additional Context**: Screenshots, examples, etc.

## 🔒 Security

If you discover security vulnerabilities:

1. **Do NOT** open a public issue
2. Email the maintainers directly
3. Provide detailed information about the vulnerability
4. Allow reasonable time for the issue to be addressed

## 📞 Getting Help

- **Issues**: Open an issue for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the README and docs/ directory
- **Examples**: Review the notebooks/ directory for usage examples

## 🎉 Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to the Real-Time Fraud Detection System! Your contributions help make financial systems more secure. 🛡️