# src/duplicate_detection.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateDetector:

    def __init__(self, df):
        self.df = df.copy()

    def detect_duplicates(self, threshold=0.95):
        """
        Flag entries that are near-duplicates based on TF-IDF cosine
        similarity of company + title + level + jobFamily text.
        """
        cols = ["company", "title", "level", "jobFamily"]

        combined = (
            self.df[cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(combined)
        similarity = cosine_similarity(matrix)

        flags = []
        for i in range(len(self.df)):
            duplicate = False
            for j in range(len(self.df)):
                if i != j and similarity[i][j] > threshold:
                    duplicate = True
                    break
            flags.append(duplicate)

        self.df["flag_duplicate"] = flags

        return self.df