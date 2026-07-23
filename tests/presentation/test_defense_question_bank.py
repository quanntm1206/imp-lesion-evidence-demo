from dataclasses import dataclass
import re
from pathlib import Path

QUESTION_BANK = Path("docs/presentation/defense-question-bank.md")


@dataclass(frozen=True)
class Question:
    number: int
    answer: str


def parse_question_bank(path: Path) -> dict[int, Question]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^### (?P<number>\d+)\. ", text)
    return {
        int(blocks[index]): Question(
            int(blocks[index]),
            blocks[index + 1].split("**Evidence:**", 1)[0].strip(),
        )
        for index in range(1, len(blocks), 2)
    }


def test_questions_keep_separate_evidence_lanes():
    questions = parse_question_bank(QUESTION_BANK)
    assert questions[4].answer.startswith("**Lane: live reconstructed lane")
    assert questions[6].answer.startswith("**Lane: fixed L206 cache")
    assert questions[18].answer.startswith(
        "**Lane: historical Paper RQ1 versus prospective RQ1-v2"
    )
