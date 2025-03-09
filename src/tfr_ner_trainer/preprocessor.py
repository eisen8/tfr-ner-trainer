import re


def preprocessor(s: str) -> str:
    # Remove websites links
    s = re.sub(r'https?:\/\/', '', s)
    s = re.sub(r'www\.([a-zA-Z0-9]+)\.([a-zA-Z0-9.\/]+)', r'', s, flags=re.IGNORECASE)  # websites with periods
    s = re.sub(r'www\s([a-zA-Z0-9]+)\s([a-zA-Z0-9.\/]+)', r'', s, flags=re.IGNORECASE)  # websites with whitespace

    # Add space between Season and Episode (i.e. S05E05 -> S05 E05) so it can be tokenized easier
    s = re.sub(r'S(\d+)E(\d+)', r'S\1 E\2', s, flags=re.IGNORECASE)
    s = re.sub(r'S(\d+)P(\d+)E(\d+)', r'S\1 P\2 E\3', s, flags=re.IGNORECASE) # season part episode
    s = re.sub(r'S(\d+)P(\d+)', r'S\1 P\2', s, flags=re.IGNORECASE) # season part
    s = re.sub(r'S(\d+)Ep(\d+)', r'S\1 E\2', s, flags=re.IGNORECASE)  # case where e is ep

    # Replaces all periods with space except if between two digits (i.e. '7.1' for sound)
    s = re.sub(r'\.(?!\d)|(?<!\d)\.', ' ', s)
    s = re.sub(r'(?<!\d\.\d)\.(?!(\d$|\d\D|\d\.\d))', ' ', s)  # cases of year/ep/resolution i.e. "1973.480p" or "E02.720p"

    # Replace _ with spaces
    s = re.sub(r'_', ' ', s)

    # Remove dashes
    s = re.sub(r'-([\[\(\]\)])', r'\g<1>', s)  # dashes before brackets
    s = re.sub(r'([\[\(\]\)])-', r'\g<1>', s)  # dash after brackets
    s = re.sub(r'(?<=[\w\[\]\(\)])(-)(?=[\w\[\]\(\)]*$)', ' ', s)  # dash at end such as x265-GalaxyTV or x264-mSD[eztv]

    # Add whitespace before non-prefix open brackets/parenthesis
    s = re.sub(r'(?<=[a-zA-Z0-9\]\)])[\[\(]', r' \g<0>', s)

    # Add whitespace after non-suffix closed brackets/parenthesis
    s = re.sub(r'[\]\)](?=[a-zA-Z0-9\[\(])', r'\g<0> ', s)

    # We allow '-', this is just to normalize them with either no spaces or spaces on both sides.
    # Without this "x264-[yts]" would become "x264- yts". This normalizes it to "x264 - yts"
    s = re.sub(r'(?<=\S) -|-(?= \S)', ' - ', s)

    # Remove empty brackets and parenthesis
    s = re.sub(r'\(\)', ' ', s)
    s = re.sub(r'\[\]', ' ', s)

    # Replace multiple spaces or dashes with a single
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'-+', '-', s)

    # Random fixes
    s = re.sub(r' x 26', ' x26', s, flags=re.IGNORECASE)  # some uploaders separate the x in x265
    s = re.sub(r' h 26', ' h26', s, flags=re.IGNORECASE)  # some uploaders separate the h in x265

    # Strip whitespace or dashes from the start and end of the string
    s = re.sub(r'^[\s-]+|[\s-]+$', '', s)

    return s