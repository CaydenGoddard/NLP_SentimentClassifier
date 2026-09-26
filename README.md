# Sentiment Classifier — Logistic Regression by Hand

A from-scratch sentiment classifier built without any machine learning libraries.
Every piece — tokenizing, feature extraction, the weighted sum, and the sigmoid
function — is implemented manually to show how logistic regression works
under the hood.

**Author:** Cayden Goddard
**Date:** 9/19/2026

## How it works

The pipeline runs in five steps:

1. **`tokenize()`** — lowercases the text and strips punctuation, splitting it
   into a list of words.
2. **`extract_features()`** — converts the token list into a numeric feature
   vector (see below).
3. **`compute_z()`** — computes the weighted sum `z = w1*x1 + w2*x2 + ... + bias`.
4. **`sigmoid()`** — squashes `z` into a probability between 0 and 1 using
   `1 / (1 + e^-z)`.
5. **`classify()`** — labels the text `"positive"` if the probability is above
   0.5, otherwise `"negative"`.

## Features

Each sentence is converted into a 5-number vector:

| # | Feature | Description |
|---|---------|-------------|
| x1 | Positive word count | Number of words matching a small positive-word set (love, great, amazing, ...) |
| x2 | Negative word count | Number of words matching a small negative-word set (hate, bad, awful, ...) |
| x3 | Exclamation mark | 1 if `!` appears in the original text, else 0 |
| x4 | Negation ("not") | 1 if the token "not" appears, else 0 |
| x5 | Intensifier | 1 if a word like "very", "really", "extremely", or "so" appears, else 0 |

## Weights

```python
weights = [1.5, -2.0, 1.0, -2.0, 0.5]
bias = 0.0
```

- **+1.5** for positive words, **-2.0** for negative words — the core signal.
- **+1.0** for `!` — treated as a mild positive/excitement signal.
- **-2.0** for negation — chosen instead of the template's suggested `-1.0`
  because `-1.0` wasn't strong enough to flip phrases like *"not good"*
  (1.5 - 1.0 = 0.5 still rounds to positive). `-2.0` correctly pushes it
  negative.
- **+0.5** for intensifiers — a small nudge.
- **Bias = 0.0** — with no evidence at all, the model predicts exactly 0.5
  probability (a coin flip).

## Running it

Requires only the Python standard library (`math`, `string`).

```bash
python3 sentiment_classifier.py
```

## Sample output

```
Sentiment Classifier Results
==================================================
Text: I love this!
Features: [1, 0, 1, 0, 0]
Probability: 0.924
Prediction: positive
--------------------------------------------------
Text: This is bad
Features: [0, 1, 0, 0, 0]
Probability: 0.119
Prediction: negative
--------------------------------------------------
Text: This is not good
Features: [1, 0, 0, 1, 0]
Probability: 0.378
Prediction: negative
--------------------------------------------------
Text: This is awesome
Features: [1, 0, 0, 0, 0]
Probability: 0.818
Prediction: positive
--------------------------------------------------
Text: I hate this product
Features: [0, 1, 0, 0, 0]
Probability: 0.119
Prediction: negative
--------------------------------------------------
Accuracy on the labeled dataset: 100%
```

The model correctly classifies all 10 labeled examples in the built-in
dataset (100% accuracy), including the tricky negation cases ("I do not like
this", "It is not great").

## Limitations

This is a hand-built demonstration model, not a trained one — the weights
were chosen manually rather than learned via gradient descent. It only
recognizes a small, fixed set of positive/negative words, so it won't
generalize well to sentences using vocabulary outside those lists.
