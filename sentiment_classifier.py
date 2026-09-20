"""
Sentiment Classifier - Logistic Regression by Hand

Name: Cayden Goddard
Date: 9/19/2026

How it works:
1. tokenize()          -> clean text into a list of lowercase words
2. extract_features()  -> turn text into a list of numbers (feature vector)
3. compute_z()         -> weighted sum:  z = w1*x1 + w2*x2 + ... + bias
4. sigmoid()           -> squash z into a probability between 0 and 1
5. classify()          -> probability > 0.5 means "positive", else "negative"
"""

import math
import string

# ---------------------------------------------------------------------------
# Word lists used to build features
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {"love", "great", "amazing", "good", "awesome",
                  "wonderful", "fantastic", "happy"}
NEGATIVE_WORDS = {"hate", "bad", "awful", "terrible", "worst",
                  "horrible", "sad"}
INTENSIFIERS = {"very", "really", "extremely", "so"}

# ---------------------------------------------------------------------------
# Small hardcoded dataset (label = what a human would say the sentence is)
# ---------------------------------------------------------------------------
DATA = [
    ("I love this product", "positive"),
    ("This is amazing and great", "positive"),
    ("I hate this", "negative"),
    ("This is awful", "negative"),
    ("This is really good!", "positive"),
    ("The worst thing ever", "negative"),
    ("Awesome, I love it!", "positive"),
    ("What a terrible, bad day", "negative"),
    ("I do not like this", "negative"),
    ("It is not great", "negative"),
]


def tokenize(text):
    """
    Convert text to lowercase and split into words.

    Example:
    "I love this!" -> ["i", "love", "this"]
    """
    # Lowercase so "Love" and "love" count as the same word
    text = text.lower()

    # Remove punctuation (.,!? etc.)
    for ch in string.punctuation:
        text = text.replace(ch, "")

    # Split into words on whitespace
    tokens = text.split()
    return tokens


def extract_features(text):
    """
    Convert text into a feature vector.

    x1 = count of positive words   (evidence FOR positive)
    x2 = count of negative words   (evidence FOR negative)
    x3 = 1 if "!" appears, else 0  (excitement, leans positive here)
    x4 = 1 if "not" appears, else 0 (negation, flips meaning, leans negative)
    x5 = 1 if an intensifier (very, really, ...) appears, else 0

    Returns:
    A list like [x1, x2, x3, x4, x5]
    """
    tokens = tokenize(text)

    positive_count = sum(1 for word in tokens if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in tokens if word in NEGATIVE_WORDS)

    # tokenize() strips punctuation, so check the ORIGINAL text for "!"
    exclamation = 1 if "!" in text else 0

    not_present = 1 if "not" in tokens else 0

    extra_feature = 1 if any(word in INTENSIFIERS for word in tokens) else 0

    features = [positive_count, negative_count, exclamation, not_present, extra_feature]
    return features


def sigmoid(z):
    """
    Apply the sigmoid function.

    Formula:
    1 / (1 + e^-z)

    Big positive z -> close to 1. Big negative z -> close to 0. z = 0 -> 0.5.
    """
    return 1 / (1 + math.exp(-z))


def compute_z(features, weights, bias):
    """
    Compute:
    z = w·x + b
    (multiply each feature by its weight, add them up, then add the bias)
    """
    z = sum(w * x for w, x in zip(weights, features)) + bias
    return z


def predict_probability(features, weights, bias):
    """
    Return probability that the text is positive.
    """
    z = compute_z(features, weights, bias)
    probability = sigmoid(z)
    return probability


def classify(probability):
    """
    Convert probability into label.

    Rule:
    > 0.5 = positive
    <= 0.5 = negative
    """
    if probability > 0.5:
        return "positive"
    else:
        return "negative"


def print_prediction(text, weights, bias):
    """
    Print prediction results.
    """
    features = extract_features(text)
    probability = predict_probability(features, weights, bias)
    label = classify(probability)

    print(f"Text: {text}")
    print(f"Features: {features}")
    print(f"Probability: {round(probability, 3)}")
    print(f"Prediction: {label}")
    print("-" * 50)


def accuracy(data, weights, bias):
    """
    Fraction of labeled sentences the model gets right.
    """
    correct = 0
    for text, true_label in data:
        features = extract_features(text)
        probability = predict_probability(features, weights, bias)
        if classify(probability) == true_label:
            correct += 1
    return correct / len(data)


def main():
    # WEIGHTS (one per feature, same order as extract_features):
    #   1.5  positive words -> positive weight = evidence FOR positive sentiment
    #  -2.0  negative words -> negative weight = evidence FOR negative sentiment
    #   1.0  "!"            -> excitement nudges toward positive
    #  -2.0  "not"          -> negation pushes toward negative. (The template
    #                          suggested -1.0, but that made "not good" come
    #                          out positive, since 1.5 - 1.0 > 0. -2.0 fixes it.)
    #   0.5  intensifier    -> small nudge toward positive
    # BIAS 0.0 means with no evidence at all, probability is exactly 0.5.
    weights = [1.5, -2.0, 1.0, -2.0, 0.5]
    bias = 0.0

    test_sentences = [
        "I love this!",
        "This is bad",
        "This is not good",
        "This is awesome",
        "I hate this product",
    ]

    print("Sentiment Classifier Results")
    print("=" * 50)

    for sentence in test_sentences:
        print_prediction(sentence, weights, bias)

    print(f"Accuracy on the labeled dataset: {accuracy(DATA, weights, bias):.0%}")


if __name__ == "__main__":
    main()