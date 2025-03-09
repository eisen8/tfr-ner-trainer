import time

import spacy
from spacy.training import offsets_to_biluo_tags

from tokenizer import custom_tokenizer

if __name__ == "__main__":
    start = time.perf_counter()  # Start high-resolution timer
    test_data = [
        "Space Force S01 E02 720p NF WEBRip x264 GalaxyTV",
        "Space Force 1x2 720p NF WEBRip x264 GalaxyTV",
        "[ENTE] Dragon Ball (1986) S 06 E 06 [AV1] [OPUS] [DVD] [480p]",
        "The Big Bang Theory S 11 E 22 720p HDTV x264 AVS [eztv]",
        "What If 2021 S 01 E 08 720p WEBRip x265 MiNX",
        "Spider Man Far From Home 2019 NEW 720p HD TC X264 1XBET",
        "Paradise 2025 S01 E01 480p x264 RUBiK",
        "Dragon Ball - S01 E10 - The Dragon Balls are Stolen!"
        "[Golumpa] Boogiepop and Others - 05 [FuniDub 720p x264 AAC][40F8E573"
    ]

    nlp = spacy.load("tfr-model")
    nlp.tokenizer = custom_tokenizer(nlp)
    model_loaded_time = time.perf_counter()  # Start high-resolution timer
    print(f"Elapsed model_load time: {model_loaded_time - start:.9f} seconds")

    for text in test_data:
        start_text = time.perf_counter()  # Start high-resolution timer
        doc = nlp(text)
        # Access tokens, named entities, and parts of speech
        print(f"---- {text}")
        tokens = [token.text for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"Tokens {tokens}")
        print(f"Entities {entities}")

        # biluo_tags = offsets_to_biluo_tags(doc, entitites)
        # print(f"BILOU Tags: {biluo_tags}")
        end_text = time.perf_counter()  # Start high-resolution timer
        print(f"Elapsed text time: {end_text - start_text:.9f} seconds")

    end = time.perf_counter()  # Start high-resolution timer
    print(f"Elapsed total time: {end - start:.9f} seconds")
