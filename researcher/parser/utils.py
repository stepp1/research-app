import json
import logging
from pathlib import Path

#TODO: support pagination using `async` parameter

def process_ascii(ascii_text):
    ascii_text = ascii_text.replace("c¸", "ç")
    ascii_text = ascii_text.replace("C¸", "Ç")
    ascii_text = ascii_text.replace("´e", "é")
    ascii_text = ascii_text.replace("´a", "á")

    return ascii_text

def get_authors_str(authors, first_author = False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output

def check_publisher(text):
    if "Published in" in text:
        return True
    if "Publisher" in text:
        return True
    if "ICLR" in text:
        return True
    if "International Conference on Learning Representations" in text:
        return True

def is_complete(line):
    if line.endswith("."):
        return True
    if line.endswith("?"):
        return True
    if line.endswith("!"):
        return True
    if line.endswith(":"):
        return False
    if line.endswith(";"):
        return False
    if line.endswith(","):
        return False
    if line.endswith(")"):
        return False
    if line.endswith("]"):
        return False
    if line.endswith("}"):
        return False
    if line.endswith("..."):
        return True
    if line.endswith("…"):
        return True
    if line.endswith("-"):
        return False
    if line.lower().endswith("deep"):
        return False
    if line.lower().endswith("machine"):
        return False
    if line.lower().endswith("surrogate"):
        return False
    else:
        return True

def is_parsed(pdf_path, out_file):
    # check if the out_file exists
    if not Path(out_file).exists():
        return False

    # check pdf_path is already in the out_file
    with open(out_file, "r") as f:
        result_json = json.load(f)
        for paper in result_json:
            if paper["file"] == str(pdf_path):
                logging.info(f"Paper already extracted")
                return True