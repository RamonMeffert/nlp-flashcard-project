from typing import Dict, List


def context_to_reader_input(result: Dict[str, List[str]]) \
        -> Dict[str, List[str]]:
    """Takes the output of the retriever and turns it into a format the reader
    understands.

    Args:
        result (Dict[str, List[str]]): The result from the retriever
    """

    # Take the number of valeus of an arbitrary item as the number of entries
    # (This should always be valid)
    num_entries = len(result['n_chapter'])

    # Prepare result
    reader_result = {
        'titles': [],
        'texts': []
    }

    for n in range(num_entries):
        # Get the most specific title
        if result['subsection'][n] != 'nan':
            title = result['subsection'][n]
        elif result['section'][n] != 'nan':
            title = result['section'][n]
        else:
            title = result['chapter'][n]

        reader_result['titles'].append(title)
        reader_result['texts'].append(result['text'][n])

    return reader_result


def remove_formulas(ds):
    """Replaces text in the 'text' column of the ds which has an average
    word length of <= 3.5 with blanks. This essentially means that most
    of the formulas are removed.
    To-do:
    - more-preprocessing
    - a summarization model perhaps
    Args:
        ds: HuggingFace dataset that contains the information for the retriever
    Returns:
        ds: preprocessed HuggingFace dataset
    """
    words = ds['text'].split()
    average = sum(len(word) for word in words) / len(words)
    if average <= 3.5:
        ds['text'] = ''
    return ds
