# 🧠 MoodLens

**Understand the emotions behind your words.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

MoodLens is a production-ready Machine Learning application that analyzes emotions from written text. Built with a clean, modular architecture and a modern Streamlit interface, it provides real-time emotion detection with visual feedback and actionable recommendations.

### 🎯 Key Features

- **Emotion Detection**: Identifies 6 emotions (Joy, Sadness, Anger, Fear, Love, Surprise)
- **Confidence Scoring**: Shows prediction confidence with visual gauge
- **Probability Distribution**: Visual breakdown of all emotion probabilities
- **Word Cloud**: Visual representation of important words in your text
- **Important Words**: Shows which words most influenced the prediction
- **Recommendations**: Actionable tips and activities based on detected emotion
- **Text Statistics**: Character, word, and sentence counts
- **Modern UI**: Clean, minimal, and responsive design

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/moodcanvas.git
cd moodcanvas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (place in data/raw/)
# Files needed: train.txt, val.txt, test.txt

# Train the model
python train.py

# Run the application
streamlit run app.py

Usage
Open your browser at http://localhost:8501

Enter your text in the text area

Click Analyze Emotion

View the results with visualizations and recommendations

📊 Model Performance
Metric	Score
Accuracy	90.4%
F1-Weighted	90.5%
F1-Macro	87.8%
Per-Class Performance
Emotion	Precision	Recall	F1-Score
Joy	0.937	0.911	0.924
Sadness	0.939	0.924	0.931
Anger	0.919	0.909	0.914
Fear	0.869	0.811	0.839
Love	0.790	0.933	0.856
Surprise	0.747	0.877	0.807
🏗️ Project Architecture
text
MoodCanvas/
├── app.py                    # Streamlit application
├── train.py                  # Model training script
├── predict.py                # Prediction module
├── config.py                 # Central configuration
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── .gitignore                # Git ignore rules
├── CHANGELOG.md              # Version history
│
├── data/
│   └── raw/                  # Dataset files
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── models/                   # Trained models
│   ├── logistic_regression.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metadata.json
│
├── utils/                    # Utility modules
│   ├── data_loader.py        # Data loading
│   ├── preprocessor.py       # Text preprocessing
│   ├── recommender.py        # Recommendation engine
│   ├── visualizer.py         # Visualization utilities
│   └── word_importance.py    # Word importance analysis
│
├── experiments/              # Experiment logs
│   ├── dataset_inspection.md
│   └── model_training_report.json
│
└── docs/                     # Documentation
    ├── architecture.md
    └── deployment.md
🛠️ Technology Stack
Component	Technology
Language	Python 3.12
Frontend	Streamlit
Machine Learning	scikit-learn
NLP	TF-IDF Vectorizer
Model	Logistic Regression
Visualization	Plotly, Matplotlib, WordCloud
Data Processing	Pandas, NumPy
Serialization	Joblib
🎨 Visualization Examples
Probability Distribution
https://assets/screenshots/probabilities.png

Confidence Gauge
https://assets/screenshots/gauge.png

Word Cloud
https://assets/screenshots/wordcloud.png

Dashboard
https://assets/screenshots/dashboard.png

🔬 Experiments
The project follows a structured experimentation approach:

Milestone 1: Dataset Inspection

Milestone 2: Configuration & DataLoader

Milestone 3: Text Preprocessing

Milestone 4: Model Training (Logistic Regression)

Milestone 5: Prediction Module

Milestone 6: Recommendation Engine

Milestone 7: Word Importance Analysis

Milestone 8: Visualization Utilities

Milestone 9: Word Cloud

Milestone 10: Streamlit Interface

Milestone 11: Documentation (Current)

🗺️ Future Roadmap
Short Term
Add more emotion classes

Implement model versioning

Add batch processing

Export results as PDF

Medium Term
Try alternative models (SVM, Random Forest)

Implement hyperparameter tuning

Add explainability (SHAP/LIME)

Multi-language support

Long Term
Deploy as web service

Add user accounts

Historical analysis dashboard

Mobile application

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch

Make your changes

Run tests

Submit a pull request

Development Setup
bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .

# Check linting
flake8
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Kaggle Emotion Dataset

scikit-learn community

Streamlit team

All open-source contributors

📧 Contact
Author: Your Name

Email: your.email@example.com

GitHub: github.com/yourusername

⭐ Show Your Support
If you found this project useful, please give it a star ⭐ on GitHub!

🧠 MoodLens - Understand the emotions behind your words.