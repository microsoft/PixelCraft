import time
import re
import os
from src.utils.model import ModelClient

DEFAULT_LMM_SYS_MSG = """You are a helpful AI assistant."""
DEFAULT_MODEL = "gpt-4-turbo"

deployment_name = os.environ.get("DEPLOYMENT_NAME", "gpt-4.1-mini-2025-04-14")
client = ModelClient().get_client()

def match_score(prompt):
    retry_limit = 10
    for _ in range(retry_limit):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            score = response.choices[0].message.content
            return int(score.strip())
        except Exception as e:
            time.sleep(1)
    return 'Failed'

def compute_acc_single(ex, pred):
    if pred is not None:
        pred = pred.replace('||', '|')
    else:
        pred = ""
    if "answer" in ex:
        solution = ex["answer"]
    elif "Answer" in ex:
        solution = ex["Answer"]
    else:
        raise ValueError(f"No answer found in the example: {ex}")
    # Use regex to find text between 'FINAL ANSWER:' and 'TERMINATE'
    result = re.search(r'FINAL ANSWER:\s*(.*?)\s*TERMINATE', pred)

    # Extract the matched text if found
    prediction = result.group(1) if result else pred
    answer_score = 0
    grading_query = f'''
    Rate my prediction given the correct answer.
    Prediction: {prediction}
    Answer: {solution}
    If you think the prediction is correct, return 1, otherwise return 0. Return 0 or 1 only.'''
    answer_score = match_score(grading_query)
    return answer_score, prediction, solution


def compute_acc_from_raw_answer(question, solution, pred):
    pred = pred.replace('||', '|')
    result = re.search(r'FINAL ANSWER:\s*(.*?)\s*TERMINATE', pred)

    # Extract the matched text if found
    prediction = result.group(1) if result else pred
    answer_score = 0
    grading_query = f'''
    You are given a prediction for the question. You need to rate it given the correct answer. Disregard the format, and only rate based on the content. If you think the prediction is correct, i.e. same as the correct answer, then return 1, otherwise return 0. Return 0 or 1 only.

    # Example
    # Question: What is the height of the tower?
    # Prediction: ANSWER: The tower is built in China from 200 years ago. The total hight of the tower is 180 Meters. TERMINATE
    # Correct Answer: 80 Meters
    # Your Response: 0

    # Example
    # Question: What is the difference?
    # Prediction: ANSWER: 69%. TERMINATE
    # Correct Answer: 69
    # Your Response: 1

    # Example
    # Question: What is the increase?
    # Prediction: ANSWER: 3%. TERMINATE
    # Correct Answer: 0.03
    # Your Response: 1

    # Example
    # Question: What is the ratio?
    # Prediction: ANSWER: 1.541. TERMINATE
    # Correct Answer: 1.54
    # Your Response: 1

    # Question: {question}
    # Prediction: {prediction}
    # Correct Answer: {solution}
    # Your Response:'''
    # print(grading_query)
    answer_score = match_score(grading_query)
    return answer_score, prediction
