from base_model.retriever import Retriever

if __name__ == '__main__':
    # Initialize retriever
    r = Retriever()

    # Retrieve example
    retrieved = r.retrieve(
        "When is a stochastic process said to be stationary?")

    for i, (score, result) in enumerate(retrieved):
        print(f"Result {i+1} (score: {score * 100:.02f}:")
        print(result['text'][0])
        print()  # Newline
