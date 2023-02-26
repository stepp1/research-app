import logging
from pathlib import Path
from turtle import pd
from typing import List

from researcher.parser import parser
from scripts.img2txt import extract_from_dataset as img2txt_dataset
from scripts.pdf2img import extract_from_dataset as pdf2img_dataset

# log to file
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def check_streamlit() -> bool:
    """
    Function to check whether python code is run within streamlit

    Returns
    -------
    use_streamlit : boolean
        True if code is run within streamlit, else False
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit


def build_from_abstract(list_with_abstracts):
    raise NotImplementedError


def build_from_url(list_with_urls):
    raise NotImplementedError


def build_from_title(list_with_titles):
    raise NotImplementedError


def build_from_pdf(list_with_pdfs):
    raise NotImplementedError


def build_dataset(
    input,
    out_dir: str = None,
    **kwargs,
):
    # check if streamlit is running
    if check_streamlit():
        out_dir = "/tmp/"

    # check if input is a dict
    if isinstance(input, dict):
        papers = input[0]
        # check if it contains title or url or abstract
        if not any(key in papers for key in ["title", "url", "abstract"]):
            raise ValueError("Invalid dataset format")

        # check if abstract exists as key
        elif "abstract" in papers:
            build_from_abstract(papers)

    # obtain papers and save a out/papers.json
    papers = parser(
        input, Path(input).is_file(), Path(input).is_dir(), out_dir=out_dir, **kwargs
    )


def dataset_pipeline(dataset: dict, out_dir: str = None, **kwargs):
    """
    Pipeline to extract PDFs context from a dataset json.
    """
    _ = pdf2img_dataset(dataset, output_folder=out_dir, **kwargs)
    return img2txt_dataset(dataset, **kwargs)


def run(
    path: str, extract_content: bool = False, out_dir: str = None, **kwargs
) -> List[dict]:
    """
    Main function to run the pipeline

    # papers = [
    # {
    #   title:    str
    #   url:      str
    #   authors:  List[str]
    #   abstract: str
    #   file:     str
    #       if extract_content:
    #           image_path: str
    #           text:       str
    # },
    # ...
    # ]

    :param path: path to the file or directory
    :param extract_content: extract images and text from pdfs (default: False)
    :param out_dir: output directory to dataset.json (default: ./ or /tmp/)
    """
    # check if streamlit is running
    if check_streamlit():
        out_dir = "/tmp/"

    # obtain papers and save a out/papers.json
    papers = parser(
        path, Path(path).is_file(), Path(path).is_dir(), out_dir=out_dir, **kwargs
    )

    # even if the path is a file, parse returns a list
    if extract_content:
        _ = pdf2img_dataset(papers, output_folder=out_dir, **kwargs)
        papers = img2txt_dataset(papers, **kwargs)

    return papers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the file or directory",
    )
    parser.add_argument(
        "--extract_content",
        type=bool,
        default=False,
        help="Extract images and text from pdfs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory to dataset.json",
    )

    run(**parser.parse_args())
