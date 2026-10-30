"""Training-free baseline: select the most common parsed final answer."""

from common import parse_answer


def predict(bank, arrays, checkpoint):
    """Return one candidate index per question, breaking ties by input order.

    ``arrays`` and ``checkpoint`` are accepted for the official runner interface.
    This baseline uses neither of them and has no inference assets.
    """
    choices = []
    for row in bank:
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts = {
            answer: answers.count(answer)
            for answer in answers if answer is not None
        }
        if counts:
            choices.append(max(range(4), key=lambda j: counts.get(answers[j], 0)))
        else:
            choices.append(0)
    return choices
