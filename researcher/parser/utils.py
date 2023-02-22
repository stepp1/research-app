import json
import logging
from pathlib import Path
from typing import Dict, List

# TODO: support pagination using `async` parameter


def process_ascii(ascii_text):
    ascii_text = ascii_text.replace("c¸", "ç")
    ascii_text = ascii_text.replace("C¸", "Ç")
    ascii_text = ascii_text.replace("´e", "é")
    ascii_text = ascii_text.replace("´a", "á")

    return ascii_text.strip()


def get_authors_str(authors, first_author=False):
    output = str()
    if first_author is False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output


def check_publisher(text):
    if "published" in text.lower():
        return True
    if "publisher" in text.lower():
        return True
    if "ICLR" in text:
        return True
    if "international conference on learning representations" in text:
        return True
    if "http" in text:
        return True
    if "Journal" in text:
        return True
    if "doi" in text:
        return True
    if "arxiv" in text.lower():
        return True
    if "Society" in text:
        return True
    if "copyright" in text.lower():
        return True
    if "p." in text.lower():
        return True
    if "pp." in text:
        return True
    if "vol." in text.lower():
        return True
    if "volume" in text.lower():
        return True
    if "article" in text.lower():
        return True
    if "open" == text.lower():
        return True
    if "open access" == text.lower():
        return True
    if "open-access" == text.lower():
        return True
    if "original research" == text.lower():
        return True
    if "review" in text.lower():
        return True
    if "sciencedirect" in text.lower():
        return True
    if "springer" in text.lower():
        return True
    if "elsevier" in text.lower():
        return True
    if "elsevier" in text.lower().replace(" ", ""):
        return True
    if "accepted" in text.lower():
        return True
    if "received" in text.lower():
        return True
    if "opinion" in text.lower():
        return True
    if "available at " in text.lower():
        return True
    if "available at:" in text.lower():
        return True
    if "Estuarine, Coastal and Shelf Science" in text:
        return True
    if "Geomorphology " in text:
        return True
    if "Hydrobiologia" in text:
        return True
    if "Marine Policy" in text:
        return True
    if "Science of the Total Environment" in text:
        return True
    if "©" in text:
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
    results_exist = Path(out_file).exists()
    if results_exist:
        # check pdf_path is already in the out_file
        result_json = json.load(open(out_file, "r"))
        for paper in result_json:
            if paper["file"] == str(pdf_path):
                logging.info(f"Paper already extracted")
                return True
    return False


def add_to_json(new_result: List[Dict], current_out_file: str):
    # check if file exists
    print(current_out_file, Path(current_out_file).exists())
    if not Path(current_out_file).exists():
        raise FileNotFoundError(f"File {current_out_file} does not exist")

    # add result to the json file if it exists
    with open(current_out_file) as f:
        results = json.load(f)

    results = results + new_result

    with open(current_out_file, "w") as f:
        json.dump(results, f, indent=4)

    return results
