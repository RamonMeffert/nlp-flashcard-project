# %%
from datasets import load_dataset
from src.retrievers.faiss_retriever import FaissRetriever


data = load_dataset("GroNLP/ik-nlp-22_slp", "paragraphs")

# # %%
# x = data["test"][:3]

# # %%
# for y in x:

#     print(y)
# # %%
# x.num_rows

# # %%
retriever = FaissRetriever(data)
scores, result = retriever.retrieve("hello world")
