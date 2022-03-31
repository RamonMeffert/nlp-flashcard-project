import gradio as gr
import pandas as pd

from query import default_probe, get_retrieval_span_scores


def spaces_probe(question: str):
    answers, scores, context = default_probe(question)

    answers_text = [answer.text for answer in answers]
    d_scores, s_scores = get_retrieval_span_scores(answers)

    formatted_result = pd.DataFrame(zip(answers_text, d_scores.tolist(), s_scores.tolist()), columns=[
        "answer", "document score", "span score"])

    formatted_result["position"] = formatted_result.index + 1

    return formatted_result


interface = gr.Interface(spaces_probe, inputs="text", outputs=["dataframe"])
interface.launch()
