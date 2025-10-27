from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Main grouping function
def create_similarity_groups(texts, labels, group_size=4, max_features=5000):
    """
    Group texts by TF-IDF + cosine similarity with balanced labels.

    Args:
        texts: List of text strings
        labels: List of labels (0 for negative, 1 for positive)
        group_size: Size of each group (default 4: 2 positive + 2 negative)
        max_features: Maximum features for TF-IDF vectorizer

    Returns:
        List of groups, each containing indices, labels, texts, and similarity score
    """
    # Separate by labels
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]

    print(f"Total texts: {len(texts)}")
    print(f"Positive reviews: {len(positive_indices)}")
    print(f"Negative reviews: {len(negative_indices)}")

    # Compute TF-IDF
    print("\nComputing TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity
    print("Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Create groups
    print("Creating similarity groups...")
    groups = []
    used_positive = set()
    used_negative = set()

    for pos_idx in positive_indices:
        if pos_idx in used_positive:
            continue

        # Find most similar positive review
        pos_similarities = [(i, cosine_sim[pos_idx][i])
                           for i in positive_indices
                           if i != pos_idx and i not in used_positive]

        if not pos_similarities:
            continue

        pos_similarities.sort(key=lambda x: x[1], reverse=True)
        second_pos_idx = pos_similarities[0][0]

        # Find two most similar negative reviews
        avg_pos_vector = (tfidf_matrix[pos_idx] + tfidf_matrix[second_pos_idx]) / 2

        neg_similarities = []
        for neg_idx in negative_indices:
            if neg_idx not in used_negative:
                sim = cosine_similarity(avg_pos_vector, tfidf_matrix[neg_idx])[0][0]
                neg_similarities.append((neg_idx, sim))

        if len(neg_similarities) < 2:
            continue

        neg_similarities.sort(key=lambda x: x[1], reverse=True)
        first_neg_idx = neg_similarities[0][0]
        second_neg_idx = neg_similarities[1][0]

        # Create group
        group = {
            'indices': [pos_idx, second_pos_idx, first_neg_idx, second_neg_idx],
            'labels': [labels[pos_idx], labels[second_pos_idx],
                      labels[first_neg_idx], labels[second_neg_idx]],
            'texts': [texts[pos_idx], texts[second_pos_idx],
                     texts[first_neg_idx], texts[second_neg_idx]],
            'avg_similarity': np.mean([
                cosine_sim[pos_idx][second_pos_idx],
                cosine_sim[first_neg_idx][second_neg_idx],
                cosine_sim[pos_idx][first_neg_idx],
                cosine_sim[pos_idx][second_neg_idx],
                cosine_sim[second_pos_idx][first_neg_idx],
                cosine_sim[second_pos_idx][second_neg_idx]
            ])
        }

        groups.append(group)
        used_positive.add(pos_idx)
        used_positive.add(second_pos_idx)
        used_negative.add(first_neg_idx)
        used_negative.add(second_neg_idx)

    return groups


# Token filtering function (not used by default)
def filter_by_token_count(texts, labels, tokenizer, max_tokens):
    """
    Filter texts by token count.

    Args:
        texts: List of text strings
        labels: List of labels
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum token count threshold

    Returns:
        filtered_texts, filtered_labels, filtered_indices
    """
    filtered_indices = []
    filtered_texts = []
    filtered_labels = []

    for idx, (text, label) in enumerate(zip(texts, labels)):
        token_count = len(tokenizer.encode(text))
        if token_count <= max_tokens:
            filtered_indices.append(idx)
            filtered_texts.append(text)
            filtered_labels.append(label)

    print(f"Original size: {len(texts)}")
    print(f"Filtered size: {len(filtered_texts)}")
    print(f"Removed: {len(texts) - len(filtered_texts)} reviews")

    return filtered_texts, filtered_labels, filtered_indices