
# MNIST Model MLOps Project

A lightweight MNIST classifier with complete MLOps pipeline including automated testing, validation, and deployment.

## Project Overview

This project implements a lightweight CNN model for MNIST digit classification with the following constraints and features:

- Model Parameters: < 25k
- Training Accuracy: > 95%
- Test Accuracy: > 95%
- Automated CI/CD Pipeline
- Model Performance Tracking
- Code Quality Checks

## Project Structure

```
project_root/
│
├── src/
│   ├── __init__.py
│   ├── model.py      # Model architecture
│   ├── train.py      # Training script
│   └── dataset.py    # Data loading utilities
│
├── tests/
│   ├── __init__.py
│   └── test_model.py # Test cases
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml  # CI/CD pipeline
│
├── setup.py          # Package setup
├── requirements.txt  # Dependencies
├── conftest.py       # Pytest configuration
└── README.md         # This file
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mnist-model-mlops.git
cd mnist-model-mlops
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate # on Windows use `.venv\Scripts\activate`
```

3. Install dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training the Model

To train the model, run the following command:

```bash
# From project root directory
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/train.py
```
or
```bash
python -m src.train
```
This will:
- Download MNIST dataset (if not present)
- Train the model for specified epochs
- Save the best model as 'best_model.pth'
- Display training progress and results

### Running Tests

To run the test suite, use the following commands:

```bash
# Run all tests
pytest tests/ -v
```

## GitHub Setup and CI/CD

1. Create a new GitHub repository.

2. Push your code to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/mnist-model-mlops.git
git push -u origin main
```

3. GitHub Actions will automatically:
   - Run tests on every push and pull request.
   - Check code formatting.
   - Generate test coverage reports.
   - Create releases with trained models (on main branch).

4. View pipeline results:
   - Go to your repository on GitHub.
   - Navigate to **Actions** tab.
   - Click on the latest run.
   - Click on the **ml-pipeline** workflow.
   - Click on the job you're interested in (e.g. tests or linting).
   - Click on the log link under "Logs".

## Model Architecture

```python
class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
        
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.1)
```

## Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** OneCycleLR (max_lr=0.01)
- **Batch Size:** 128
- **Epochs:** Configurable (default=1)

## Test Cases

### Model Architecture:
- Parameter count < 25k
- Correct output shape

### Model Performance:
- Training accuracy > 95%
- Test accuracy > 95%

## License

Distributed under the MIT License. See LICENSE for more information.
