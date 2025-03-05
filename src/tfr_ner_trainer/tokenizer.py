# Define custom infix pattern for splitting between numbers and letters
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

infix_patterns = [
    r'\s+',        # Whitespace
    r'-',          # Dash
    r'(?<=\d)[xX](?=\d)',  # 1x2
    r'(?<=\d)[vV](?=\d)',  # 1v2
]

prefix_patterns = [r'\[', r'\(', r'{', r',', r'[sS](?=\d+)', r'[eE](?=\d+)', r'E[pP](?=\d+)', r'[pP](?=\d+)', r'Part(?=\d+)', r'BD']
suffix_patterns = [r'\]', r'\)', r'}', r',', r'th', r'rd', r'nd']

suffix_re = compile_infix_regex(suffix_patterns)
prefix_re = compile_infix_regex(prefix_patterns)
infix_re = compile_infix_regex(infix_patterns)

def custom_tokenizer(nlp):
    # The default tokenizer with added custom infix patterns
    tok = Tokenizer(nlp.vocab)
    tok.infix_finditer = infix_re.finditer
    tok.prefix_search = prefix_re.search
    tok.suffix_search = suffix_re.search
    return tok