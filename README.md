---
title: Speech_Language_Processing_Jurafsky_Martin
emoji: 📚
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 2.9.0
python_version: 3.10.4
app_file: app.py
pinned: true
---


# NLP FlashCards


## DEMO

View the demo at huggingface spaces:

[DEMO](https://huggingface.co/spaces/RugNlpFlashcards/Speech_Language_Processing_Jurafsky_Martin)


## Dependencies

Make sure you have the following tools installed:

- [Python](https://www.python.org/downloads/) ^3.10,<3.11
- [Poetry](https://python-poetry.org/) for Python package management;
- [Docker](https://www.docker.com/get-started/) for running ElasticSearch.
- [Git LFS](https://git-lfs.github.com/) for downloading binary files that do not fit in git.

Then, run the following commands to install dependencies and Elasticsearch:

```sh
poetry install
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.1.1
docker network create elastic
docker run --name es01 --net elastic -p 9200:9200 -p 9300:9300 -it docker.elastic.co/elasticsearch/elasticsearch:8.1.1
```

After the last command, a password for the `elastic` user should show up in the
terminal output (you might have to scroll up a bit). Copy this password, and
create a copy of the `.env.example` file and rename it to `.env`. Replace the
`<password>` placeholder with your copied password. The .env file can be used to change configuration of the system, leave it as is for a replication study.

Next, run the following command **from the root of the repository**:

```sh
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
```

**NOTE 1:** If docker is not available or feasable. It is possible to use a trail hosted version of Elasticsearch at: https://www.elastic.co/cloud/

**NOTE 2** Installing dependencies without poetry is possible, but it is not our recommendation. To do so execute `pip install -r requirements.txt`

## Running

To make sure we're using the dependencies managed by Poetry, run `poetry shell`
before executing any of the following commands. Alternatively, replace any call
like `python file.py` with `poetry run python file.py` (but we suggest the shell
option, since it is much more convenient).

### Using the QA system

⚠️ **Important** ⚠️ _If you want to run an ElasticSearch query, make sure the
docker container is running! You can check this by running `docker container
ls`. If your container shows up (it's named `es01` if you followed these
instructions), it's running. If not, you can run `docker start es01` to start
it, or start it from Docker Desktop._

To query the QA system, run any query as follows:

```sh
python query.py "Why can dot product be used as a similarity metric?"
```

By default, the best answer along with its location in the book will be
returned. If you want to generate more answers (say, a top-5), you can supply
the `--top=5` option. The default retriever uses [FAISS](https://faiss.ai/), but
you can also use [ElasticSearch](https://www.elastic.co/elastic-stack/) using
the `--retriever=es` option. You can also pick a language model using the
`--lm` option, which accepts either `dpr` (Dense Passage Retrieval) or
`longformer`. The language model is used to generate embeddings for FAISS, and
is used to generate the answer.

### CLI overview

To get an overview of all available options, run `python query.py --help`. The
options are also printed below.

```sh
usage: query.py [-h] [--top int] [--retriever {faiss,es}] [--lm {dpr,longformer}] str

positional arguments:
  str                   The question to feed to the QA system

options:
  -h, --help            show this help message and exit
  --top int, -t int     The number of answers to retrieve
  --retriever {faiss,es}, -r {faiss,es}
                        The retrieval method to use
  --lm {dpr,longformer}, -l {dpr,longformer}
                        The language model to use for the FAISS retriever
```


### Replicating the experiment

To fully run experiments, you need to run the following command:

```
# in the root of the project and poetry environment activated
python main.py
```

This command run all questions trough the system and stores the output to the `results/` directory.

After performing the experiment, results can be analyzed and displayed by running `plot.py` and the `results/*_analysis.ipynb` files.
