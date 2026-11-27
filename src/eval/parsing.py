"""Turning model output text into answers, and deciding what counts as a format failure.

Format failures (abstract.md section 7) are reported for every model and benchmark and count as wrong
in the main score. The rules are deliberately simple and written down here so they can be quoted in the report.

ScienceQA   the output must name a valid option letter (A, B, C, ...). Accepted forms:
            "B", "B.", "B. text", "(B)", "B)", "Answer: B", "The answer is B", or "B" followed by a newline.
            A lowercase letter is accepted only when it is the whole answer ("b", "b."), since otherwise it is
            usually the article "a". Not accepted: a letter that starts an ordinary sentence ("A cat is ..."),
            a letter beyond the number of options, anything else.
            (lmms-eval accepts only "B", "b" and "B. text"; ours is a superset, so it never scores lower.)
POPE        the output must begin with "yes" or "no" (case-insensitive, leading punctuation ignored).
TextVQA     the output is a format failure if it is empty or longer than 10 words.
"""

import re

_ANSWER_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is|:)\s*", re.IGNORECASE)
_LETTER = re.compile(r"^[\(\[]?([A-Z])(?:[\)\]]|[.:,\n]|$)")
_LONE_LOWER = re.compile(r"^[\(\[]?([a-z])[\)\]]?\.?$")
_YES_NO = re.compile(r"^[\W_]*(yes|no)\b", re.IGNORECASE)


def parse_choice(text, n_choices):
    """Option index (0-based) named by `text`, or None if the output has no valid option letter."""
    text = _ANSWER_PREFIX.sub("", text.strip())
    m = _LETTER.match(text) or _LONE_LOWER.match(text)
    if not m:
        return None
    idx = ord(m.group(1).upper()) - ord("A")
    return idx if idx < n_choices else None


def parse_yes_no(text):
    """'yes', 'no', or None if the output does not begin with either."""
    m = _YES_NO.match(text)
    return m.group(1).lower() if m else None


def is_textvqa_format_failure(text):
    words = text.split()
    return len(words) == 0 or len(words) > 10
