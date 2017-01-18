#!/usr/bin/env python3
# N A I V E   B A Y E S   C L A S S I F I E R

# Project   Naive-Bayes Classifier
# Author    Barnabas Markus
# Email     barnabasmarkus@gmail.com
# Date      18.01.2017
# Python    3.6
# License   MIT


from collections import Counter, defaultdict
from operator import mul
from functools import reduce


class NaiveBayes:

    def __init__(self):
        self.words_cats = defaultdict(lambda: defaultdict(int))
        self.cats = defaultdict(int)
        self.thresholds = {}

    @staticmethod
    def get_words(doc):
        """Return list of words from str"""
        return [word.lower().strip() for word in doc.split() if len(word) > 2]

    def train(self, doc, cat):
        """Train the classifier with document-category pairs"""
        words = self.get_words(doc)
        for word in words:
            self.words_cats[word][cat] += 1
            self.cats[cat] += 1

    def word_probability(self, word, cat):
        """Return probability of a word belongs to a given category"""
        if word not in self.words_cats:
            return 0
        return self.words_cats[word][cat] / sum(self.words_cats[word].values())

    def weighted_word_probability(self, word, cat, weight=1.0, ap=0.5):
        # ap: asssumed_probability
        # weight: the weight of assumed_probability
        probability = self.word_probability(word, cat)
        totals = sum(self.words_cats[word].values())
        return ((weight * ap) + (totals * probability)) / (weight + totals)

    def doc_probability(self, doc, cat):
        """Return probability of a doc belongs to a given category"""
        words = self.get_words(doc)
        weighted_probabilites = [self.weighted_word_probability(word, cat)
                                 for word in words]
        return reduce(mul, weighted_probabilites, 1)

    def cat_probability(self, cat):
        """Return probability of a category compared to all categories"""
        return self.cats[cat] / sum(self.cats.values())

    def probability(self, doc, cat):
        cat_probability = self.cat_probability(cat)
        doc_probability = self.doc_probability(doc, cat)
        return cat_probability * doc_probability

    def set_threshold(self, cat, threshold):
        """Set thredhold for category"""
        self.thresholds[cat] = threshold

    def get_threshold(self, cat):
        """Get threshold of category"""
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    def classify(self, doc, default=None):
        """Classifing a document"""
        results = {}
        for cat in self.cats:
            results[cat] = self.probability(doc, cat)

        # best_1, best_2 = Counter(results).most_common(2)
        (cat_1, score_1), (_, score_2) = Counter(results).most_common(2)
        threshold = self.get_threshold(cat_1)
        
        if score_1 < score_2 * threshold:
            return default
        return cat_1
