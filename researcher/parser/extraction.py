import io
import logging

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from researcher.parser.utils import check_publisher, is_complete, process_ascii


class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pagenos = set()
        self.rsrcmgr = PDFResourceManager(caching=True)
        self.outfp = sys.stdout if False else io.StringIO()
        self.laparams = LAParams()
        self.device = TextConverter(self.rsrcmgr, self.outfp, laparams=self.laparams)

        self.title = ""
        self.title_line = -1

        self.next_line = ""

    def get_metadata(self):
        for page in PDFPage.get_pages(self.current_fh, self.pagenos, caching=True, check_extractable=True):
            self.interpreter.process_page(page)
            text = self.outfp.getvalue()

            for i, line in enumerate(text.splitlines()):
                if len(line) > 2 and not check_publisher(line):
                    line = process_ascii(line)
                    
                    if is_complete(line) and self.title_line == -1:
                        self.title_line = i
                        self.title += line
                        logging.info(f"Possible title: {self.title}")
                        continue
                    
                    if self.title_line != -1 and len(line) > 2:
                        self.next_line += line
                        logging.info(f"Next line: {self.next_line}")
                        break
            break

    def extract(self):
        with open(self.pdf_path, "rb") as fh:
            self.current_fh = fh
            self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)
            
            self.get_metadata()

        self.device.close()
        self.outfp.close()
        return self.title, self.next_line