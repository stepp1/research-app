# Extract images from a folder with PDFs, PDF file or dataset card
# Author: Stepp1
# Date: 2023
# License: GPL

import json
import logging
from pathlib import Path
from typing import List, Union

import pdf2image
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def extract_from_pdf(pdf_path: Union[str, Path], output_folder=None, prefix=None):
    """Extract images from PDF and save them to output_folder"""

    output_folder = Path(pdf_path).parent if output_folder is None else output_folder
    prefix = Path(pdf_path).name.split(".")[0] if prefix is None else prefix

    images = pdf2image.convert_from_path(
        pdf_path,
        dpi=150,
        fmt="jpg",
        output_folder=output_folder,
        thread_count=10,
        hide_annotations=True,
        output_file=prefix,
    )
    return images


def extract_from_folder(folder: Union[str, Path], output_folder=None, prefix=None):
    """Extract images from PDFs in a folder and save them to output_folder"""

    base_output_folder = Path(folder).parent if output_folder is None else output_folder

    all_images = []
    for pdf in tqdm(Path(folder).glob("*.pdf")):
        output_folder = Path(base_output_folder) / "jpg" / Path(pdf.stem)
        output_folder.mkdir(exist_ok=True, parents=True)

        images = extract_from_pdf(pdf, output_folder, prefix)
        all_images.extend(images)

    return all_images


def extract_from_dataset(ds_card: dict, output_folder=None) -> List[str]:
    """Extract images from PDFs in a dataset card and save them to output_folder"""
    all_images = []
    for data in tqdm(ds_card):
        # print(data["file"], data["image_path"])
        if not Path(data["file"]).exists():
            print(f"File not found: {data['file']}")
            continue

        # if image_path is not specified, create correct path
        # researcher/data/jpg/ Path(data["file"]).stem
        if "image_path" not in data and output_folder is None:
            data["image_folder"] = str(
                Path(data["file"]).parent.parent / "jpg" / Path(data["file"]).stem
            )
        elif "image_path" not in data and output_folder is not None:
            data["image_folder"] = str(
                Path(output_folder) / "jpg" / Path(data["file"]).stem
            )

        # remove old images
        for image in Path(data["image_path"]).glob("*.jpg"):
            image.unlink()

        images = extract_from_pdf(data["file"], data["image_path"])

        # set images list
        data["images"] = [str(image.filename) for image in images]

        all_images.extend(images)

    return all_images


if __name__ == "__main__":
    """
    Extract images from a folder with PDFs, PDF file or dataset card

    Usage:
        python pdf2img.py -d <dataset_card.json>
        python pdf2img.py -i <pdf_path>
        python pdf2img.py -f <folder_with_pdfs>

    """
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from images")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset card",
    )
    parser.add_argument(
        "-i",
        "--pdf",
        type=str,
        help="pdf path",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="folder with PDFs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output folder",
        default=None,
    )

    args = parser.parse_args()
    if args.pdf:
        images = extract_from_pdf(args.pdf, args.output)
        logging.info(f"Done! Check {args.output}")
        exit(0)

    elif args.folder:
        images = extract_from_folder(args.folder, args.output)
        logging.info(f"Done! Check {args.output}")
        exit(0)

    else:
        assert args.dataset, "Please specify dataset card, image or folder."

    ds_card = json.load(open(args.dataset))
    images = extract_from_dataset(ds_card)

    logging.info(f"Done!")

    compress_bash = (
        "tar --exclude='pdf/*.pdf' -cvf - **/* | xz -v -6 -z - > data.tar.xz"
    )
    logging.info(f"Compress with: {compress_bash}")
