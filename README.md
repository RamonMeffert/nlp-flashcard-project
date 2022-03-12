# nlp-flashcard-project

## Todo voor progress meeting

- [ ] Data inlezen/Repo klaarmaken
- [ ] Proof of concept met UnifiedQA
- [ ] Standaard QA model met de dataset
- [ ] Papers verzamelen/lezen
- [ ] Eerder werk bekijken, inspiratie opdoen voor research richting

## Overview

De meeste QA systemen bestaan uit twee onderdelen:

- Een retriever. Die haalt adhv de vraag _k_ relevante stukken context op, bv.
  met `tf-idf`.
- Een model dat het antwoord genereert. Wat je hier precies gebruikt hangt af
  van de manier van question answering:
  - Voor **extractive QA** gebruik je een reader;
  - Voor **generative QA** gebruik je een generator.

  Beide werken op basis van een language model.

## Handige info

- Huggingface QA tutorial: <https://huggingface.co/docs/transformers/tasks/question_answering#finetune-with-tensorflow>
- Overview van open-domain question answering technieken: <https://lilianweng.github.io/posts/2020-10-29-odqa/>

## Base model

Tot nu toe alleen een retriever die adhv een vraag de top-k relevante documents
ophaalt. Haalt voor veel vragen wel hoge similarity scores, maar de documents
die die ophaalt zijn meestal niet erg relevant.

```bash
poetry shell
cd base_model
poetry run python main.py
```

### Voorbeeld

"What is the perplexity of a language model?"

> Result 1 (score: 74.10):  
> Figure 10 .17 A sample alignment between sentences in English and French, with
> sentences extracted from Antoine de Saint-Exupery's Le Petit Prince and a
> hypothetical translation. Sentence alignment takes sentences e 1 , ..., e n ,
> and f 1 , ..., f n and finds minimal > sets of sentences that are translations
> of each other, including single sentence mappings like (e 1 ,f 1 ), (e 4 -f 3
> ), (e 5 -f 4 ), (e 6 -f 6 ) as well as 2-1 alignments (e 2 /e 3 ,f 2 ), (e 7
> /e 8 -f 7 ), and null alignments (f 5 ).
>
> Result 2 (score: 74.23):  
> Character or word overlap-based metrics like chrF (or BLEU, or etc.) are
> mainly used to compare two systems, with the goal of answering questions like:
> did the new algorithm we just invented improve our MT system? To know if the
> difference between the chrF scores of two > MT systems is a significant
> difference, we use the paired bootstrap test, or the similar randomization
> test.
>
> Result 3 (score: 74.43):  
> The model thus predicts the class negative for the test sentence.
>
> Result 4 (score: 74.95):  
> Translating from languages with extensive pro-drop, like Chinese or Japanese,
> to non-pro-drop languages like English can be difficult since the model must
> somehow identify each zero and recover who or what is being talked about in
> order to insert the proper pronoun.
>
> Result 5 (score: 76.22):  
> Similarly, a recent challenge set, the WinoMT dataset (Stanovsky et al., 2019)
> shows that MT systems perform worse when they are asked to translate sentences
> that describe people with non-stereotypical gender roles, like "The doctor
> asked the nurse to help her in the > operation".
