# Extract text from images using easyocr
# Author: Stepp1
# Date: 2023
# License: GPL


import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Union

import easyocr
import numpy as np
import pandas as pd
import PIL
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# log to file
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def open_img(path: Union[str, Path]) -> PIL.Image:
    """Open image and convert to grayscale and resize to 1000x1200"""
    return PIL.Image.open(path).convert("L").resize((1000, 1200))


def img2txt(img_path: Union[str, Path], **kwargs) -> str:
    """Extract text from image"""
    logging.info(f"Processing {str(img_path)}")
    image = np.array(open_img(img_path))

    result = reader.readtext(image, detail=0, paragraph=True, **kwargs)
    result = " ".join(result)
    return result


def img2txt_batched(batch_path: List[Union[str, Path]], batch_size=16, **kwargs) -> str:
    logging.info(f"Processing {str(batch_path)}")
    images = [np.array(open_img(path)) for path in batch_path]

    result = reader.readtext_batched(
        images,
        detail=0,
        paragraph=True,
        batch_size=batch_size,
        n_height=1200,
        n_width=1000,
        **kwargs,
    )

    for i in range(len(result)):
        result[i] = " ".join(result[i])

    result = " ".join(result)
    return result


def list_images(path: Union[str, Path]) -> List[Path]:
    """List all images in a folder"""
    images_paths = []
    for img_pth in Path(path).glob("*.jpg"):
        images_paths.append(img_pth)

    return sorted(images_paths)


def extract_from_dataset(
    dataset: Dict[str, Union[List[str], str]]
) -> Dict[str, Union[List[str], str]]:
    """Extract text from images in dataset card"""

    for i, sample in enumerate(dataset):
        sample_text_path = Path(sample["image_folder"]) / "text.parquet"

        logging.info(f"Processing {i}th card. {sample['title']}")
        start = time.time()

        if sample_text_path.exists():
            logging.info(f"{sample_text_path} exists.")
            table = pq.read_table(str(sample_text_path))
            df = table.to_pandas()
            sample["text"] = df["text"].values[0]
            logging.info(f"Done in {time.time() - start} seconds")
            continue

        image_paths = list_images(sample["image_folder"])
        batch_size = 16  # max in RTX 3090
        n_batches = len(image_paths) // batch_size
        logging.info(f"Number of batches: {n_batches}")

        text = ""
        for j in range(n_batches):
            start_batch = time.time()
            batch = image_paths[j * batch_size : (j + 1) * batch_size]
            text += img2txt_batched(batch, batch_size)
            logging.info(f"Batch {j} done in {time.time() - start_batch} seconds")

        sample["text"] = text
        logging.info(f"Done in {time.time() - start} seconds")

        # create dataframe from sample dictionary
        df = pd.DataFrame([sample])
        table = pa.Table.from_pandas(df)

        pq.write_table(table, str(sample_text_path))
        logging.info(f"Saved to {sample_text_path}")

    return dataset


reader = easyocr.Reader(["en"], gpu=True, quantize=True)
# pytorch 2.0
torch.compile(reader.detector)
torch.compile(reader.recognizer)

if __name__ == "__main__":
    """
    Extract text from image in a folder, specified image or dataset card.

    Usage:
        python img2txt.py -d <dataset_card.json>
        python img2txt.py -i <image_path>
        python img2txt.py -f <folder_with_images>

    """
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from image_folder")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset card",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="image path",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="folder with image_folder",
    )

    args = parser.parse_args()
    if args.image:
        out_text = img2txt(args.image)
        print(out_text)
        sys.exit(0)

    elif args.folder:
        out_text = img2txt_batched(list_images(args.folder))
        print(out_text)
        sys.exit(0)

    else:
        assert args.dataset, "Please specify dataset card, image or folder."

    ds_card = json.load(open(args.dataset))
    ds_card = extract_from_dataset(ds_card)

    output_path = Path(args.dataset).parent / "full_dataset.json"

    with open(output_path, "w") as f:
        json.dump(ds_card, f)

    output_parquet_path = Path(args.dataset).parent / "full_dataset.parquet"
    out_df = pd.DataFrame(ds_card)
    out_table = pa.Table.from_pandas(out_df)
    pq.write_table(out_table, str(output_parquet_path))

    logging.info("Done!")
