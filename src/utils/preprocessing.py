from typing import Dict, List


def result_to_reader_input(result: Dict[str, List[str]]) \
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
