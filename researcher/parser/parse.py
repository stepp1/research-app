import argparse
import json
import sys
from pathlib import Path
from time import sleep
from typing import Dict, List, Union

import arxiv
from fuzzywuzzy import fuzz

from researcher.parser.extraction import PDFExtractor
from researcher.parser.scraper import serpapi_scrape_google_scholar_organic_results
from researcher.parser.utils import add_to_json, get_authors_str, is_parsed

# https://serpapi.com/
api_key = "3549e959aaba6263113445e812dbc67dbf961422e5cfc9d109e28d9103d54be0"
use_serpapi = False

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def arxiv_search(query: str):
    logging.info(f"Querying arxiv with = {query}")
    query_results = arxiv.Search(
        query=query,
        max_results=1,
    )
    try:
        for result in query_results.results():
            paper_id = result.get_short_id()
            paper_title = result.title
            paper_url = result.entry_id
            paper_abstract = result.summary.replace("\n", " ")
            paper_authors = result.authors
            paper_first_author = get_authors_str(result.authors, first_author=True)
            primary_category = result.primary_category
            publish_time = result.published.date()
            update_time = result.updated.date()
            comments = result.comment

    except Exception as e:
        logging.error(f"Error while parsing arxiv results: {e}")
        return None

    paper = {
        "id": paper_id,
        "title": paper_title,
        "url": paper_url,
        "abstract": paper_abstract,
        "authors": paper_authors,
        "first_author": paper_first_author,
    }

    return paper


def serpapi_search(query: str):
    logging.info(f"Querying G.Scholar using serpapi with = {query} + arxiv")
    result = serpapi_scrape_google_scholar_organic_results(
        query=query + "+ arxiv",
        api_key=api_key,
        lang="en",
    )[0]

    if fuzz.ratio(result["title"], query) < 65:
        logging.info(f"Querying G.Scholar using serpapi with = {query}")
        result = serpapi_scrape_google_scholar_organic_results(
            query=query,
            api_key=api_key,
            lang="en",
        )[0]

    if "arxiv" in result["link"]:
        logging.info(f"Found arxiv link: {result['link']}")
        arxiv_id = result["link"].split("/")[-1]
        return arxiv_search(arxiv_id)

    paper_title = result["title"]
    paper_url = result["link"]
    try:
        paper_authors = [
            author["name"] for author in result["publication_info"]["authors"]
        ]

        paper_first_author = get_authors_str(paper_authors, first_author=True)
    except KeyError:
        paper_authors = [""]
        paper_first_author = ""
    paper_abstract = result["snippet"]

    paper = {
        "id": arxiv_id if "arxiv" in result["link"] else result["link"],
        "title": paper_title,
        "url": paper_url,
        "abstract": paper_abstract,
        "authors": paper_authors,
        "first_author": paper_first_author,
    }

    return paper


def get_paper_metadata(
    title: str, next_line: str = "", try_google: bool = True
) -> Dict[str, Union[str, List[str]]]:
    """
    Get the title, URL, authors and abstract of a paper using either arxiv or Google Scholar

    :param title: the title of the paper
    :param next_line: the next line of the paper title

    :return: a dictionary with the keys "title", "url", and "authors"
    """
    query = title + " " + next_line

    if not use_serpapi:
        paper = arxiv_search(query)

    else:
        paper = serpapi_search(query)

    if fuzz.ratio(title, paper["title"]) < 60 and try_google:
        logging.warning(f"Title mismatch {title} vs {paper['title']}")
        paper = serpapi_search(query)
    elif fuzz.ratio(title, paper["title"]) < 60:
        logging.warning("Title mismastch Skipping...")
        return None

    logging.info(
        f"Title: {str(paper['title'])}, URL: {str(paper['url'])}, Authors: {get_authors_str(paper['authors'])}"
    )

    out_dict = {
        "title": str(paper["title"]),
        "url": str(paper["url"]),
        "authors": [str(author) for author in paper["authors"]],
        "abstract": str(paper["abstract"]),
    }
    return out_dict


def extract_paper(
    pdf_path: str, out_file: str, **kwargs
) -> Union[None, Dict[str, Union[str, List[str]]]]:
    """
    This function extracts paper information from a pdf file.
    :param pdf_path: Path to the pdf file
    :param out_file: Output file path

    :return: None if the pdf file is already parsed or a dictionary containing paper information if not parsed yet.
    """
    logging.info(f"Extracting paper from {pdf_path}")

    if is_parsed(pdf_path, out_file):
        logging.info("Already parsed")
        return None

    extractor = PDFExtractor(pdf_path)
    title, next_line = extractor.extract()
    logging.info(f"Title: {title}, Next line: {next_line}")
    try:
        paper = get_paper_metadata(title.lower(), next_line=next_line.lower(), **kwargs)
        paper["file"] = str(pdf_path)
        logging.info("Confirmed")

    except Exception as e:
        paper = None
        logging.error(f"Error: {e}")

    return paper


def parse_dir(
    dir_path: str, out_file: str, **kwargs
) -> List[Dict[str, Union[str, List[str]]]]:
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
            paper = extract_paper(file, out_file, **kwargs)
            if paper is None:
                logging.info(f"Skipping {file}")
                continue

            papers.append(paper)

        sleep(2)  # avoid getting blocked
    return papers


def parser(
    path: str,
    is_file: bool = False,
    is_dir: bool = False,
    out_dir: str = "./",
    **kwargs,
) -> Union[None, List[Dict]]:
    """
    This function parses either a single pdf file or all pdf files in a directory.
    :param path: Path to either a pdf file or a directory
    :param is_file: Flag to indicate if the path is a pdf file
    :param is_dir: Flag to indicate if the path is a directory
    :param out_dir: Path to the output directory

    :return: None if the pdf file is already parsed or a list of one or more dictionaries containing paper information from all pdf files in the directory if a directory is parsed.
    """
    # to save the papers in a json file
    out_dir = Path(out_dir) / Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    current_out_file = out_dir / "papers.json"

    if is_file:
        logging.info(f"Extracting paper from {path}")
        papers = [extract_paper(path, current_out_file, **kwargs)]
    elif is_dir:
        logging.info(f"Extracting papers from directory {path}")
        papers = parse_dir(path, current_out_file, **kwargs)

    if papers is None:
        return None

    if Path(current_out_file).exists():
        add_to_json(papers, current_out_file)
    else:
        json.dump(papers, open(current_out_file, "w"), indent=4)

    return papers


def main(arg: argparse.Namespace) -> None:
    """
    Main function to parse the pdf file or directory path

    Arguments:
    arg (argparse.Namespace): Namespace containing the command line arguments passed to the script

    Returns:None
    """
    # Check if the path is a file or a directory
    if Path(arg.pdf_path).is_file():
        parser(arg.pdf_path, is_file=True)
    elif Path(arg.pdf_path).is_dir():
        parser(arg.pdf_path, is_dir=True)
        parser(arg.pdf_path, is_dir=True)
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
