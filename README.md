# NLP FlashCards


## DEMO

View the demo at huggingface spaces:


## Dependencies

Make sure you have the following tools installed:

- [Poetry](https://python-poetry.org/) for Python package management;
- [Docker](https://www.docker.com/get-started/) for running ElasticSearch.
- [Git LFS](https://git-lfs.github.com/) for downloading binary files that do not fit in git. 

Then, run the following commands:

```sh
poetry install
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.1.1
docker network create elastic
docker run --name es01 --net elastic -p 9200:9200 -p 9300:9300 -it docker.elastic.co/elasticsearch/elasticsearch:8.1.1
```

After the last command, a password for the `elastic` user should show up in the
terminal output (you might have to scroll up a bit). Copy this password, and
create a copy of the `.env.example` file and rename it to `.env`. Replace the
`<password>` placeholder with your copied password.

Next, run the following command **from the root of the repository**:

```sh
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
```

## Running

To make sure we're using the dependencies managed by Poetry, run `poetry shell`
before executing any of the following commands. Alternatively, replace any call
like `python file.py` with `poetry run python file.py` (but we suggest the shell
option, since it is much more convenient).

### Training

N/A for now

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
the `--retriever=es` option.

### CLI overview

To get an overview of all available options, run `python query.py --help`. The
options are also printed below.

```sh
usage: query.py [-h] [--top int] [--retriever {faiss,es}] str

positional arguments:
  str                   The question to feed to the QA system

options:
  -h, --help            show this help message and exit
  --top int, -t int     The number of answers to retrieve
  --retriever {faiss,es}, -r {faiss,es}
                        The retrieval method to use
```
