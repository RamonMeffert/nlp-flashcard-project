from retriever import Retriever


if __name__ == '__main__':
    # Initialize retriever
    r = Retriever()

    # Retrieve example
    scores, result = r.retrieve(
        "What is the perplexity of a language model?")

    for i, score in enumerate(scores):
        print(f"Result {i+1} (score: {score:.02f}):")
        print(result['text'][i])
        print()  # Newline

    # Compute overall performance
    exact_match, f1_score, total = r.evaluate()
    print(f"Exact match: {exact_match} / {total}\n"
          f"F1-score: {f1_score:.02f}")
