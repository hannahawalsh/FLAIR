"""
Downloading and parsing Financial Literacy PDFs to create a text dataset for
the Long Form Question Answering model for the Financial Literacy AI Resouce
(FLAIR)
"""
# Imports
import re
import os
from tika import parser
from collections import defaultdict
import numpy as np
import urllib.request

# Download file to local machine
def download_file(download_url, filename):
    """ Download a PDF file to the subdirectory data_sources """
    # Create the subdirectory if it doesn't exist
    pdf_dir = "data_sources"
    if not os.path.exists(pdf_dir):
        os.mkdir(pdf_dir)

    # Download and save the pdf
    response=urllib.request.urlopen(download_url)
    save_path = os.path.join(pdf_dir, filename)
    with open(save_path, "wb") as f:
        f.write(response.read())
    print(f"Saved PDF to {save_path}")


# Create a parser class
class parsePDF:
    def __init__(self, url, filename):
        self.filename = filename
        self.url = url


    def extract_contents(self, start_regex=None, download=False):
        """ Extract a pdf's contents using tika. """
        # Check if the file is downloaded
        pdf_path = os.path.join("data_sources", self.filename)
        if not os.path.exists(pdf_path):
            # Download the file to the data_sources folder
            if download == True:
                download_file(self.url, self.filename)
            else:
                # Read directly from the website without downloading
                pdf_path = self.url

        # Get raw parsed pdf text
        pdf = parser.from_file(pdf_path)
        self.text = pdf["content"]
        if "title" in pdf["metadata"]:
            self.title = pdf["metadata"]["title"]
        else:
            self.title = self.filename.split(".")[0]

        # Cut text to start at starting_chars
        if start_regex:
            try:
                split = re.split(start_regex, self.text, 1)[-1]
                self.text = start_regex.replace("\s+", "\n") + split
            except:
                print(f"Failed to find the phrase '{starting_chars}'")
        return self.text


    def clean_text(self, specific=None):
        """ Clean the raw text of the pdf to be readable English. """
        # Remove non ASCII characters: curved quotations
        self.text = re.sub(r'“|”', '"', self.text)  # double quotes
        self.text = re.sub(r'’|‘', "'", self.text)  # single quotes

        # Remove Lines starting with 
        self.text = re.sub(r"^.*$", "", self.text, flags=re.M)

        # Remove non ASCII characters: bullet points as lists
        pattern = r"\s*([•·∙⋅◦‣§■◊])\s*(.*?)\s*\1"
        self.text = re.sub(pattern, r"\n\2;", self.text, flags=re.DOTALL)

        # Remove all other non ASCII characters
        self.text = re.sub(r"[^\x00-\x7f]", "", self.text)

        # Format numbered lists as semicolon-separated lists
        self.text = re.sub(r"^[0-9]+\s*\.(.*)\n", r"\1;", self.text, flags=re.M)

        # If a line ends with a comma, remove it for clarity
        self.text = re.sub(r",(\n)+", r"\1", self.text)

        # Remove numbers from footnotes
        self.text = re.sub(r"^[0-9]+\s+(.*)", r"\1", self.text, flags=re.M)

        # Replace tabs with spaces
        self.text = re.sub(r"\t+", r" ", self.text)

        # Remove space before punctuation
        self.text = re.sub(r" ([.;,?!])", r"\1", self.text)

        # Remove numbers at end of words (usually footnotes)
        self.text = re.sub(r"(?<=[A-z.,])[0-9]+[\b\s]", " ", self.text)

        # Remove any words containing both letters and numbers
        pattern = r"[A-z]+[0-9]\S*|[0-9]+[A-z]+\S*"
        self.text = re.sub(pattern, "", self.text)

        # Do formatting on lines
        clean_text = " "
        for line in re.split(r"\n", self.text):
            line = line.strip()
            # Signals the end of the document, start of metadata
            if re.match(r"^SPREAD WITH", line):
                break

            # Specific to PDF
            if specific == "cfpb_ymyg":  # Your Money, Your Goals
                # Ignore page footers
                if re.match(r"^MODULE [0-9]+:.*[0-9]+", line):
                    continue
                elif re.match(r"[0-9]+ [A-Z]+", line):
                    continue

                # Weird example tables start with '.'
                elif line.startswith("."):
                    clean_text += "\n"
                    continue

                # Those tables seem to be surrounded by 'CATEGORY'
                elif line == "CATEGORY":
                    clean_text += "\n"
                    continue

            # End the previous sentence if extra blank line
            if not line:
                L = clean_text[-1]
                period = "." if L.isalnum() else ""
                clean_text += f"{period} \n"
                continue

            # Remove page numbers
            elif re.match(r"^[0-9\s]+$", line):
                continue

            # Remove likely footers
            elif re.search(r"[0-9]*[A-Z\s]+$", line):
                continue

            # Remove lines that are URLs
            elif re.match(r"^https?:\/\/.*[\r\n]*", line, flags=re.M):
                continue

            # Remove lines with no alphanumeric characters
            elif not re.search(r"[A-Za-z0-9]", line):
                continue

            # Remove any input fields
            elif "_" in line or ("check box" in line.lower() and ":" in line):
                continue

            # Keep Title Case headers as their own line
            elif line.istitle():
                clean_text += f"\n{line}\n"
                continue

            # Add line to text
            usespace = "" if clean_text[-1] == "\n" else " "
            clean_text += (usespace + line)
        self.text = clean_text.strip()


        # If 3 or more spaces, make it a new line
        self.text = re.sub(r"   +", "\n", self.text, flags=re.M)

        # Replace multiple whitespace with one newline
        self.text = re.sub(r"\n\s+", "\n", self.text, flags=re.M)

        # Replace multiple periods and single period lines
        self.text = re.sub(r"\.\.+", "", self.text, flags=re.M)
        self.text = re.sub(r"\n\.", "", self.text, flags=re.M)

        # Replace all other duplicate punctuation with the first one
        self.text = re.sub(r"([!,-.:;?])([!,-.:;?]+)", r"\1", self.text)

        # Make sure there's a whitespace after punctuation
        pattern = r"([A-Za-z][!,.:;?])([A-Za-z])"
        self.text = re.sub(pattern, r"\1 \2", self.text)

        # Remove whitespace after a slash (e.g., within a url)
        self.text = re.sub(r"/\s", "/", self.text)

        # Remove whitespace around hyphens
        self.text = re.sub(r"\s{0,1}-\s{0,1}", "-", self.text)

        # Specific to pdf
        if specific == "cfpb_ymyg":
            # Sometimes on pages they interject contacts for problems
            pattern = r"Having a problem.*?-[0-9]{4}\."
            self.text = re.sub(pattern, "", self.text, flags=re.DOTALL)

            # Format Module titles
            new_text = ""
            split_text = re.split(r"MODULE [0-9]+", self.text)[1:]
            for i, T in enumerate(split_text):
                # Section title
                pattern = r".*?\s?(?=[A-Z][a-z]+\s([a-z,']+\s){3})"
                title =  re.search(pattern, T.strip(), flags=re.M|re.DOTALL)
                title = title.group(0).strip().replace("\n", " ")
                new_text += f"MODULE {i+1}: {title}\n"

                # Text of section
                new_text += T.replace(title, "", 1).strip()
            self.text = new_text

        return self.text


    def get_text_data(self, chapter_regex=None, starting_id=0):
        """
        Break the text into medium-sized paragraph chunks, about 100
        words each. Then format the data into the correct format
        """
        assert hasattr(self, "text"), "You haven't parsed the text yet"
        # Split text up by 'chapter'
        if chapter_regex:
            split_text = re.split(chapter_regex, self.text)[1:]
            chapter_names = [x.group(0) for x in
                re.finditer(chapter_regex, self.text)]
            text_dict = {c: t for c, t in zip(chapter_names, split_text)}
        else:
            text_dict = {"main": self.text}

        # For each chapter, break text up into paragraphs
        self.chunks = defaultdict(list)
        for chapter_name, chapter_text in text_dict.items():
            chunk_list = self.chunks[chapter_name]
            for para in chapter_text.split("\n"):
                # Get paragraph length and previous chunk length
                paragraph_words = len(para.split())
                if len(chunk_list) > 0:
                    previous_words = len(chunk_list[-1].split())
                else:
                    previous_words = 0

                # Decide if it should be appended to previous chunk
                if not previous_words or previous_words > 100:
                    chunk_list.append(para.strip())
                elif previous_words + paragraph_words < 200:
                    chunk_list[-1] += " " + para.strip()
                else:
                    chunk_list.append(para.strip())

        # Format the data
        self.datapoints = []
        current_id = starting_id
        for section, text_list in self.chunks.items():
            for text in text_list:
                if text:
                    self.datapoints.append({
                        "_id": current_id,
                        "source_title": self.title,
                        "source_filename": self.filename,
                        "source_url": self.url,
                        "section_title": section,
                        "passage_text": text
                    })
                current_id += 1

        return self.datapoints
