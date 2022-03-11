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
