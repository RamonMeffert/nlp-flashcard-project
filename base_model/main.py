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
