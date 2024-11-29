# IR-SCORE: Interventional Radiology Safety and Complication Outcome Risk Evaluation

A machine learning-based system for predicting complications in interventional radiology procedures using synthetic data. IR-SCORE provides both predictive analytics and a practical clinical scoring system for risk assessment.

## Overview

IR-SCORE consists of three integrated components:

1. **Synthetic Data Generation Module**
2. **Machine Learning Prediction Model**
3. **Clinical Risk Scoring System**

## Requirements

- Python (>= 3.10)
- NumPy (>= 1.24)
- pandas (>= 2.0)
- scikit-learn (>= 1.3)
- matplotlib (>= 3.7)
- seaborn (>= 0.12)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/IR-SCORE.git

# Navigate to project directory
cd IR-SCORE

# Create and activate virtual environment using venv
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate synthetic dataset and run full analysis
python main.py
```

Or run components individually:

```bash
# Generate synthetic data
python interventional_oncology_predictor.py

# Run ML analysis
python ir_analysis.py

# Generate clinical risk score
python ir_score_tool.py
```

## Project Structure

```
IR-SCORE/
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── ml_analysis.py
│   └── scoring_tool.py
├── tests/
│   └── test_models.py
├── image/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── requirements.txt
├── main.py
├── Model Performance.md
└── README.md
```

## Scoring System

**Risk factors and weights:**

```python
risk_factors = {
    'age_over_65': 1,
    'albumin_below_3.5': 2,
    'inr_above_1.2': 2,
    'bmi_over_30': 1,
    'tumor_size_over_3cm': 1,
    'multiple_tumors': 1,
    'ecog_score_ge_2': 2
}
```

**Risk Categories:**

- Low Risk: 0-3 points
- Moderate Risk: 4-6 points
- High Risk: 7-10 points

## Model Details

```python
model_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'class_weight': 'balanced',
    'random_state': 42
}
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 src/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

```bibtex
@software{ir_score_2024,
  author = {Shakthi Kumaran Ramasamy},
  title = {IR-SCORE: Interventional Radiology Safety and Complication Outcome Risk Evaluation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/shakthiramasamy/IR-SCORE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Shakthi Kumaran Ramasamy] - [hi@shakthiramasamy.com], [shakthi@stanford.edu]

Project Link: [https://github.com/shakthiramasamy/IR-SCORE](https://github.com/shakthiramasamy/IR-SCORE)
