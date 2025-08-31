# gaze-quality-ml

**gaze-quality-ml** is a Python machine learning project for analyzing eye-tracking data quality and fixation performance.  
It uses structured eye-tracking datasets (e.g., fixation-level and participant-level spreadsheets) to train and evaluate models that classify or predict the reliability of gaze data.

## Features
- **Data processing**: Load and clean fixation and participant-level data from spreadsheets.  
- **Machine learning models**: Built with [scikit-learn](https://scikit-learn.org), including classification and regression approaches.  
- **Statistical analysis**: Generate quality metrics to evaluate gaze reliability.  
- **Export results**: Participant- and fixation-level summaries for further analysis.

## Tech Stack
- Python 3.x  
- scikit-learn  
- pandas  
- NumPy  

## Getting Started
Clone the repository and install the dependencies:

```bash
git clone https://github.com/jessewashburn/gaze-quality-ml.git
cd gaze-quality-ml
pip install -r requirements.txt
