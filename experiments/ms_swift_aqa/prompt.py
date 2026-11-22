"""Shared prompt for AVQA supervision and MMAU evaluation."""
from string import ascii_uppercase


def format_question(question, choices, *, allow_empty_options=False):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Expected a nonempty question')
    if not isinstance(choices, (tuple, list)) or not 2 <= len(choices) <= len(ascii_uppercase):
        raise ValueError('Expected 2–26 answer options')
    if not all(isinstance(text, str) and (text.strip() or allow_empty_options) for text in choices):
        raise ValueError('Expected nonempty text for every option')
    options = [f'{letter}. {answer}' for letter, answer in zip(ascii_uppercase, choices)]
    return '\n'.join(['<audio>', question, *options, 'Reply with the letter of the correct option.'])
