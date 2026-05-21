# IMDb-Sentiment-Analysis

Movie reviews often contain subjective opinions that can be classified as positive or negative. This project builds a sentiment analysis system that classifies IMDb movie reviews using a neural network model trained on the IMDb Movie Reviews Dataset.

## Project contents

- `sentiment_analysis.py`: training, evaluation, and prediction script
- `requirements.txt`: Python dependencies
- `.gitignore`: ignores model output and temporary files
- `tests/test_sentiment_analysis.py`: basic unit tests for key functions

## Usage

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Train and evaluate the model:

```bash
python sentiment_analysis.py --train --evaluate
```

3. Predict sentiment for a custom review:

```bash
python sentiment_analysis.py --predict "That film was exciting and emotional."
```

## Notes

- The script uses the built-in IMDb dataset from `tensorflow.keras.datasets.imdb`.
- The model is saved to `model/` by default.
- You can adjust training parameters with `--epochs` and `--batch-size`.
 - The model is saved to `model/` by default. The trainer exports a TensorFlow SavedModel directory
	 when `--model-dir` points to a directory (recommended). To save the native Keras format use a
	 filename with a `.keras` or `.h5` extension, e.g. `--model-dir model.keras`.
 - Run the tests using unittest discovery from the project root:

```bash
python -m unittest discover -s tests -p "*.py"
```
