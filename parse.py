import argparse
import json
import sys
from pathlib import Path
from time import sleep
from typing import Dict, List, Union

import arxiv
from extraction import PDFExtractor
from scraper import serpapi_scrape_google_scholar_organic_results
from utils import get_authors_str, is_parsed

# https://serpapi.com/
api_key = "3549e959aaba6263113445e812dbc67dbf961422e5cfc9d109e28d9103d54be0"
use_serpapi = False

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def get_paper_metadata(title: str, next_line: str = "") -> Dict[str, Union[str, List[str]]]:
    """
    Get the title, URL, authors and abstract of a paper using either arxiv or Google Scholar

    :param title: the title of the paper
    :param next_line: the next line of the paper title

    :return: a dictionary with the keys "title", "url", and "authors"
    """

    if not use_serpapi:
        query = title + " " + next_line
        logging.info(f"Querying arxiv with = {query}")
        query_results = arxiv.Search(
            query=query,
            max_results=1,
        )
        for result in query_results.results():
            paper_id            = result.get_short_id()
            paper_title         = result.title
            paper_url           = result.entry_id
            paper_abstract      = result.summary.replace("\n"," ")
            paper_authors       = result.authors
            paper_first_author  = get_authors_str(result.authors,first_author = True)
            primary_category    = result.primary_category
            publish_time        = result.published.date()
            update_time         = result.updated.date()
            comments            = result.comment

        
    else:
        logging.info(f"Querying G.Scholar with = {title}")
        result = serpapi_scrape_google_scholar_organic_results(
            query=title,
            api_key=api_key,
            lang="en",
        )[0]

        paper_title = result["title"]
        paper_url = result["link"]
        paper_authors = [author["name"] for author in result["publication_info"]["authors"]]
        paper_first_author = get_authors_str(paper_authors, first_author = True)

    logging.info(f"Title: {str(paper_title)}, URL: {str(paper_url)}, Authors: {get_authors_str(paper_authors)}")

    out_dict = {
        "title": paper_title,
        "url": paper_url,
        "authors": [
            str(author) for author in paper_authors
        ],
        "abstract": paper_abstract
    }
    return out_dict



def extract_paper(pdf_path: str, out_file: str) -> Union[None, Dict[str, Union[str, List[str]]]]:
    """
    This function extracts paper information from a pdf file.
    :param pdf_path: Path to the pdf file
    :param out_file: Output file path

    :return: None if the pdf file is already parsed or a dictionary containing paper information if not parsed yet.
    """
    logging.info(f"Extracting paper from {pdf_path}")

    if is_parsed(pdf_path, out_file):
        return None 

    extractor = PDFExtractor(pdf_path)

    title, next_line = extractor.extract()
                        
    result = get_paper_metadata(title.lower(), next_line=next_line.lower())
    result["file"] = str(pdf_path)

    logging.info(f"Confirmed")
    return result


def parse_dir(dir_path: str, out_file: str) -> List[Dict[str, Union[str, List[str]]]]:
    """
    This function parses all pdf files in a directory.

    Arguments:
    :param dir_path: Path to the directory
    :param out_file: Output file path

    :return: A list of dictionaries containing paper information from all pdf files in the directory.
    """
    papers = []
    for file in Path(dir_path).iterdir():
        if file.suffix == ".pdf":
            paper = extract_paper(file, out_file)
            papers.append(paper)

        sleep(2)
    return papers

def parser(path: str, is_file: bool = False, is_dir: bool = False) -> Union[None, List[Dict]]:
    """
    This function parses either a single pdf file or all pdf files in a directory.
    :param path: Path to either a pdf file or a directory
    :param is_file: Flag to indicate if the path is a pdf file
    :param is_dir: Flag to indicate if the path is a directory
    
    :return: None if the pdf file is already parsed or a list of one or more dictionaries containing paper information from all pdf files in the directory if a directory is parsed.
    """
    # to save the result in a json file
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    current_out_file = out_dir / "result.json"

    if is_file:
        logging.info(f"Extracting paper from {path}")
        result = [extract_paper(path, current_out_file)]
    
    elif is_dir:
        logging.info(f"Extracting papers from directory {path}")
        result = parse_dir(path, current_out_file)


    # add result to the json file if it exists
    if Path(current_out_file).exists():
        with open(current_out_file, "rw") as f:
            result_json = json.load(f)
            result_json.extend(result)
            json.dump(result_json, f, indent=4)
    else:        
        with open(current_out_file, "w") as f:
            json.dump(result, f, indent=4)

    return result

def main(args) -> None:
    """Main function to parse the pdf file or directory path

    Arguments:
    args (argparse.Namespace): Namespace containing the command line arguments passed to the script

    Returns:None
    """
    # Check if the path is a file or a directory
    if Path(args.pdf_path).is_file():
        parser(args.pdf_path, is_file = True)
    elif Path(args.pdf_path).is_dir():
        parser(args.pdf_path, is_dir = True)
    else:
        logging.error(f"Path {args.pdf_path} is not a file or a directory")
        sys.exit(1)

    return 

if __name__ == "__main__":
    # receive the path of the pdf file as a argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-pdf_path", "-f", help="The path to the pdf file.", type=str, default="data"
    )

    args = arg_parser.parse_args()

    main(args)

   
