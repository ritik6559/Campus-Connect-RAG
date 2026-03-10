import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Optional

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.juit.ac.in"

FACULTY_URLS: dict[str, str] = {
    "Computer Science & IT": (
        f"{BASE_URL}/computer-science-engineering-information-technology-faculty"
    ),
    "Electronics & Communication": (
        f"{BASE_URL}/electronics-communication-engineering-faculty"
    ),
    # "Humanities & Social Sciences": (
    #     f"{BASE_URL}/humanities-social-sciences-faculty"
    # ),
    # "Biotechnology & Informatics": (
    #     f"{BASE_URL}/biotechnology-and-informatics-faculty"
    # ),
    # "Civil Engineering": (
    #     f"{BASE_URL}/civil-engineering-faculty"
    # ),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

FIELD_LABELS = [
    "Faculty Name",
    "Email",
    "Contact No.",
    "Designation",
    "Date of Joining",
    "Highest Qualification",
    "Awarded Year",
]

class FacultyScraper:
    def __init__(self, delay: float = 1.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:

        try:

            logger.info(f"GET  {url}")
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            time.sleep(self.delay)

            return BeautifulSoup(r.text, "lxml")
        except requests.RequestException as exc:
            logger.error(f"FAIL {url} — {exc}")
            return None

    @staticmethod
    def _clean(text: str) -> str:
        
        return re.sub(r"\s+", " ", text).strip() if text else ""

    def _find_faculty_ul(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """
        Find the <ul> containing faculty <li> cards.
        """

        h1 = soup.find("h1")

        if h1:
            for sibling in h1.find_next_siblings():
                if sibling.name == "ul":
                    li = sibling.find("li")
                    if li and li.find("img") and "Faculty Name" in li.get_text():
                        return sibling

        for ul in soup.find_all("ul"):
            li = ul.find("li")
            if li and li.find("img") and "Faculty Name" in li.get_text():
                return ul

        logger.warning("Could not locate faculty <ul>")

        return None

    def _extract_fields(self, li: BeautifulSoup) -> dict:
        """
        Parse raw text of a <li> into a dict.
        """

        raw = li.get_text(separator="\n")
        lines = [self._clean(l) for l in raw.splitlines() if self._clean(l)]

        fields: dict[str, str] = {}
        i = 0

        while i < len(lines):
            line = lines[i]

            if " : " in line:
                label, _, value = line.partition(" : ")
                label = self._clean(label)
                value = self._clean(value)
                if label in FIELD_LABELS:
                    fields[label] = value
                    i += 1
                    continue

            if line in FIELD_LABELS and i + 1 < len(lines):
                nxt = lines[i + 1]
                if nxt == ":":
                    fields[line] = self._clean(lines[i + 2]) if i + 2 < len(lines) else ""
                    i += 3
                else:
                    fields[line] = self._clean(nxt.lstrip(": "))
                    i += 2
                continue

            i += 1

        return fields

    def _li_to_document(self, li: BeautifulSoup, department: str) -> Optional[Document]:
        fields = self._extract_fields(li)

        name = fields.get("Faculty Name", "").strip()
        if not name:
            return None

        email         = fields.get("Email", "")
        phone         = fields.get("Contact No.", "")
        designation   = fields.get("Designation", "")
        date_joined   = fields.get("Date of Joining", "")
        qualification = fields.get("Highest Qualification", "")
        awarded_year  = fields.get("Awarded Year", "")

        img = li.find("img")
        photo_url = urljoin(BASE_URL, img["src"]) if img and img.get("src") else ""

        profile_url = ""
        for a in li.find_all("a", href=True):
            if "faculty.php" in a["href"]:
                profile_url = urljoin(BASE_URL, a["href"])
                break

        parts = [f"Name: {name}", f"Department: {department}"]
        if designation:   parts.append(f"Designation: {designation}")
        if qualification: parts.append(f"Highest Qualification: {qualification}")
        if awarded_year:  parts.append(f"Qualification Details: {awarded_year}")
        if date_joined:   parts.append(f"Date of Joining: {date_joined}")
        if email:         parts.append(f"Email: {email}")
        if phone:         parts.append(f"Phone: {phone}")
        if profile_url:   parts.append(f"Profile URL: {profile_url}")

        page_content = "\n".join(parts)

        metadata = {
            "name":          name,
            "department":    department,
            "designation":   designation,
            "qualification": qualification,
            "awarded_year":  awarded_year,
            "date_joined":   date_joined,
            "email":         email,
            "phone":         phone,
            "photo_url":     photo_url,
            "profile_url":   profile_url,
            "source":        f"{BASE_URL} — {department}",
        }

        metadata = {k: str(v) for k, v in metadata.items() if v}

        return Document(page_content=page_content, metadata=metadata)

    def _scrape_page(self, soup: BeautifulSoup, department: str) -> list[Document]:

        faculty_ul = self._find_faculty_ul(soup)

        if faculty_ul is None:
            return []

        docs: list[Document] = []
        seen: set[str] = set()

        for li in faculty_ul.find_all("li", recursive=False):
            doc = self._li_to_document(li, department)
            if doc is None:
                continue
            name = doc.metadata.get("name", "")
            if name in seen:
                continue
            seen.add(name)
            docs.append(doc)

        logger.info(f"{len(docs)} faculty in [{department}]")

        return docs

    def scrape_all(self) -> list[Document]:
        """
        Scrape 2 JUIT department pages.
        """

        all_docs: list[Document] = []

        for dept, url in FACULTY_URLS.items():
            logger.info(f"\n{'─' * 60}")
            logger.info(f"Dept : {dept}")
            logger.info(f"URL  : {url}")
            soup = self._get_soup(url)

            if soup is None:
                logger.warning(f"Skipping {dept}")
                continue

            all_docs.extend(self._scrape_page(soup, dept))

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Total faculty scraped: {len(all_docs)}")

        return all_docs

if __name__ == "__main__":

    import json

    scraper = FacultyScraper(delay=1.5)
    docs = scraper.scrape_all()

    output = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    with open("faculty_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(output)} faculty → faculty_data.json")
    
    for d in output[:2]:
        print("\n--- Sample ---")
        print(d["page_content"])