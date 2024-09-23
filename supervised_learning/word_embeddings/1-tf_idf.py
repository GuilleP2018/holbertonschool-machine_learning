#!/usr/bin/env python3
"""Thios module contains a function that creates a TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """This function creates a TF-IDF embedding
    """
    vectorize = TfidfVectorizer(vocabulary=vocab)
    matrix = vectorize.fit_transform(sentences)
    embeddings = matrix.toarray()
    features = vectorize.get_feature_names_out()
    return embeddings, features
