import itertools
import random
import re
import time
from typing import Callable, Dict, List
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
from selectolax.lexbor import LexborHTMLParser
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium_stealth import stealth
from serpapi import GoogleScholarSearch


def parse(parser: Callable, organic_results_data: Callable):
    """
    Arugments:
    - parser:  Lexbor parser from scrape_google_scholar_organic_results() function.
    - organic_results_data: List to append data to. List origin location is scrape_google_scholar_organic_results() function. Line 104.

    This function parses data from Google Scholar Organic results and appends data to a List.

    It's used by scrape_google_scholar_organic_results().

    It returns nothing as it appends data to `organic_results_data`,
    which appends it to `organic_results_data` List in the scrape_google_scholar_organic_results() function.
    """

    for result in parser.css(".gs_r.gs_or.gs_scl"):
        try:
            title: str = result.css_first(".gs_rt").text()
        except:
            title = None

        try:
            title_link: str = result.css_first(".gs_rt a").attrs["href"]
        except:
            title_link = None

        try:
            publication_info: str = result.css_first(".gs_a").text()
        except:
            publication_info = None

        try:
            snippet: str = result.css_first(".gs_rs").text()
        except:
            snippet = None

        try:
            # if Cited by is present in inline links, it will be extracted
            cited_by_link = "".join(
                [
                    link.attrs["href"]
                    for link in result.css(".gs_ri .gs_fl a")
                    if "Cited by" in link.text()
                ]
            )
        except:
            cited_by_link = None

        try:
            # if Cited by is present in inline links, it will be extracted and type cast it to integer
            cited_by_count = int(
                "".join(
                    [
                        re.search(r"\d+", link.text()).group()
                        for link in result.css(".gs_ri .gs_fl a")
                        if "Cited by" in link.text()
                    ]
                )
            )
        except:
            cited_by_count = None

        try:
            pdf_file: str = result.css_first(".gs_or_ggsm a").attrs["href"]
        except:
            pdf_file = None

        organic_results_data.append(
            {
                "title": title,
                "title_link": title_link,
                "publication_info": publication_info,
                "snippet": snippet if snippet else None,
                "cited_by_link": f"https://scholar.google.com{cited_by_link}"
                if cited_by_link
                else None,
                "cited_by_count": cited_by_count if cited_by_count else None,
                "pdf_file": pdf_file,
            }
        )


def scrape_google_scholar_organic_results(
    query: str,
    pagination: bool = False,
    operating_system: str = "Windows" or "Linux",
    save_to_csv: bool = False,
    save_to_json: bool = False,
) -> List[Dict[str, str]]:
    """
    Extracts data from Google Scholar Organic resutls page:
    - title: str
    - title_link: str
    - publication_info: str
    - snippet: str
    - cited_by_link: str
    - cited_by_count: int
    - pdf_file: str

    Arguments:
    - query: str. Search query.
    - pagination: bool. Enables or disables pagination.
    - operating_system: str. 'Windows' or 'Linux', Checks for operating system to either run Windows or Linux verson of chromedriver

    Usage:
    data = scrape_google_scholar_organic_results(query='blizzard', pagination=False, operating_system='win') # pagination defaults to False

    for organic_result in data:
        print(organic_result['title'])
        print(organic_result['pdf_file'])
    """

    # selenium stealth
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # checks for operating system to either run Windows or Linux verson of chromedriver
    # expects to have chromedriver near the runnable file
    if operating_system is None:
        raise Exception(
            'Please provide your OS to `operating_system` argument: "Windows" or "Linux" for script to operate.'
        )

    if operating_system.lower() == "windows" or "win":
        driver = webdriver.Chrome(
            options=options, service=Service(executable_path="chromedriver.exe")
        )

    if operating_system.lower() == "linux":
        driver = webdriver.Chrome(
            options=options, service=Service(executable_path="chromedriver")
        )

    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    page_num: int = 0
    organic_results_data: list = []

    if pagination:
        while True:
            # parse all pages
            driver.get(
                f"https://scholar.google.com/scholar?q={query}&hl=en&gl=us&start={page_num}"
            )
            parser = LexborHTMLParser(driver.page_source)

            parse(parser=parser, organic_results_data=organic_results_data)

            # pagination
            if parser.css_first(
                ".gs_ico_nav_next"
            ):  # checks for the "Next" page button
                page_num += 10  # paginate to the next page
                time.sleep(random.randint(1, 3))  # sleep between paginations
            else:
                break
    else:
        # parse single, first page
        driver.get(
            f"https://scholar.google.com/scholar?q={query}&hl=en&gl=us&start={page_num}"
        )
        parser = LexborHTMLParser(driver.page_source)

        parse(parser=parser, organic_results_data=organic_results_data)

    if save_to_csv:
        pd.DataFrame(data=organic_results_data).to_csv(
            "google_scholar_organic_results_data.csv", index=False, encoding="utf-8"
        )
    if save_to_json:
        pd.DataFrame(data=organic_results_data).to_json(
            "google_scholar_organic_results_data.json", index=False, orient="records"
        )
    driver.quit()
    return organic_results_data


def serpapi_scrape_google_scholar_organic_results(
    query: str, api_key: str = None, lang: str = "en", pagination: bool = False
):
    """
    This function extracts data from organic results. With or without pagination.

    Arguments:
    - query: search query
    - api_key: SerpApi api key, https://serpapi.com/manage-api-key
    - lang: language for the search, https://serpapi.com/google-languages
    - pagination: True of False. Enables pagination from all pages.

    Usage:

    data = serpapi_scrape_google_scholar_organic_results(query='minecraft', api_key='serpapi_api_key', pagination=True)

    print(data[0].keys()) # show available keys

    for result in data:
        print(result['title'])
        # get other data
    """

    if api_key is None:
        raise Exception(
            "Please enter a SerpApi API key to a `api_key` argument. https://serpapi.com/manage-api-key"
        )

    if api_key and query is None:
        raise Exception(
            "Please enter a SerpApi API key to a `api_key`, and a search query to `query` arguments."
        )

    params = {
        "api_key": api_key,  # serpapi api key: https://serpapi.com/manage-api-key
        "engine": "google_scholar",  # serpapi parsing engine
        "q": query,  # search query
        "hl": lang,  # language
        "start": 0,  # first page. Used for pagination: https://serpapi.com/google-scholar-api#api-parameters-pagination-start
    }

    search = GoogleScholarSearch(params)  # where data extracts on the backend

    if pagination:
        organic_results_data = []

        while True:
            results = search.get_dict()  # JSON -> Python dict

            if "error" in results:
                print(results["error"])
                break

            organic_results_data.append(results["organic_results"])

            # check for `serpapi_pagination` and then for `next` page
            if "next" in results.get("serpapi_pagination", {}):
                search.params_dict.update(
                    dict(
                        parse_qsl(urlsplit(results["serpapi_pagination"]["next"]).query)
                    )
                )
            else:
                break

        # flatten list
        return list(itertools.chain(*organic_results_data))
    else:
        # remove page number key from the JSON response
        params.pop("start")

        search = GoogleScholarSearch(params)
        results = search.get_dict()

        if "error" in results:
            raise Exception(results["error"])

        return results["organic_results"]
