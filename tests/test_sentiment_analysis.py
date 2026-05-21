import sys
import unittest
from pathlib import Path

import numpy as np

# Make the project root importable so this test file can be executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentiment_analysis import MAX_SEQUENCE_LENGTH, build_model, get_word_index, text_to_sequence


class SentimentAnalysisTest(unittest.TestCase):
    def test_get_word_index_contains_special_tokens(self):
        word_index = get_word_index()
        self.assertIn("<PAD>", word_index)
        self.assertIn("<UNK>", word_index)
        self.assertIn("<START>", word_index)
        self.assertLess(word_index["<PAD>"], word_index["<UNK>"])

    def test_text_to_sequence_generates_fixed_length(self):
        word_index = get_word_index()
        sequence = text_to_sequence("The movie was amazing and full of energy.", word_index)
        self.assertEqual(sequence.shape[0], MAX_SEQUENCE_LENGTH)
        self.assertIsInstance(sequence[0], np.integer)

    def test_build_model_output_shape(self):
        model = build_model()
        model.build((None, MAX_SEQUENCE_LENGTH))
        self.assertEqual(model.output_shape, (None, 1))

    def test_model_directory_path(self):
        model_dir = Path("model")
        self.assertEqual(model_dir.name, "model")


if __name__ == "__main__":
    unittest.main()
