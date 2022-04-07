# %%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv("results/timings.csv", index_col="Unnamed: 0")
data
# %%
data.columns
# %%

data_retrieve = data[["faiss_dpr.retrieve", "faiss_longformer.retrieve",
                      "es_dpr.retrieve", "es_longformer.retrieve"]]

# %%
plt.title("Retrieval time")
plt.ylabel("Time (s)")
plt.xlabel("Model")
plt.boxplot(data_retrieve, labels=[
            "A1", "A2", "B1", "B2"])
plt.savefig("results/retrieval_time.png")

# %%
print(data_retrieve.describe())

with open("results/retrieval_time.tex", "w") as f:
    f.write(data_retrieve.describe().to_latex())

# %%

# now the same for the reader
data_read = data[["faiss_dpr.read", "faiss_longformer.read",
                  "es_dpr.read", "es_longformer.read"]]

plt.title("Reading time")
plt.ylabel("Time (s)")
plt.xlabel("Model")
plt.boxplot(data_read, labels=["A1", "A2", "B1", "B2"])
plt.savefig("results/read_time.png")

# %%
print(data_read.describe())

with open("results/read_time.tex", "w") as f:
    f.write(data_read.describe().to_latex())


# Statistical tests for reading time

# %%
stats.probplot(data_retrieve["es_longformer.retrieve"], dist="norm", plot=plt)
# %%


# %%
anova_retrieve = stats.f_oneway(*data_retrieve.T.values)
anova_read = stats.f_oneway(*data_read.T.values)

print(f"retrieve\n {anova_retrieve} \n\nread\n {anova_read}")

# %%
