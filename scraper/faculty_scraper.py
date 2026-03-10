import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Optional
import os

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
    "Humanities & Social Sciences": (
        f"{BASE_URL}/humanities-social-sciences-faculty"
    ),
    "Biotechnology & Informatics": (
        f"{BASE_URL}/biotechnology-and-informatics-faculty"
    ),
    "Civil Engineering": (
        f"{BASE_URL}/civil-engineering-faculty"
    ),
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
        h1 = soup.find("h1")
        if not h1:
            logger.warning("No <h1> found on page")
            return None

        for sibling in h1.find_next_siblings():
            if sibling.name == "ul":
                sample_li = sibling.find("li")
                if sample_li and sample_li.find("img"):
                    logger.debug("Found faculty <ul> via h1 sibling walk")
                    return sibling
                
        for ul in soup.find_all("ul"):
            li = ul.find("li")
            if li and li.find("img") and "Faculty Name" in li.get_text():
                logger.debug("Found faculty <ul> via fallback scan")
                return ul

        logger.warning("Could not locate faculty <ul>")
        return None

    def _extract_fields(self, li: BeautifulSoup) -> dict:
        
        raw = li.get_text(separator="\n")
        lines = [self._clean(l) for l in raw.splitlines() if self._clean(l)]

        fields: dict[str, str] = {}

        i = 0
        while i < len(lines):
            line = lines[i]

            if " : " in line:
                parts = line.split(" : ", 1)
                label = self._clean(parts[0])
                value = self._clean(parts[1])
                if label in FIELD_LABELS:
                    fields[label] = value
                    i += 1
                    continue

            if line in FIELD_LABELS and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line == ":":
                    value = lines[i + 2] if i + 2 < len(lines) else ""
                    fields[line] = self._clean(value)
                    i += 3
                    continue
                elif not next_line.startswith("http"):
                    fields[line] = self._clean(next_line.lstrip(": "))
                    i += 2
                    continue

            i += 1

        return fields

    def _li_to_document(self, li: BeautifulSoup, department: str) -> Optional[Document]:
        fields = self._extract_fields(li)

        name = fields.get("Faculty Name", "").strip()
        if not name:
            return None

        email          = fields.get("Email", "")
        phone          = fields.get("Contact No.", "")
        designation    = fields.get("Designation", "")
        date_of_join   = fields.get("Date of Joining", "")
        qualification  = fields.get("Highest Qualification", "")
        awarded_year   = fields.get("Awarded Year", "")

        img = li.find("img")
        photo_url = urljoin(BASE_URL, img["src"]) if img and img.get("src") else ""

        profile_url = ""
        for a in li.find_all("a", href=True):
            href = a["href"]
            if "faculty.php" in href:
                profile_url = urljoin(BASE_URL, href)
                break

        page_content_parts = [
            f"Name: {name}",
            f"Department: {department}",
        ]
        if designation:
            page_content_parts.append(f"Designation: {designation}")
        if qualification:
            page_content_parts.append(f"Highest Qualification: {qualification}")
        if awarded_year:
            page_content_parts.append(f"Qualification Details: {awarded_year}")
        if date_of_join:
            page_content_parts.append(f"Date of Joining: {date_of_join}")
        if email:
            page_content_parts.append(f"Email: {email}")
        if phone:
            page_content_parts.append(f"Phone: {phone}")
        if profile_url:
            page_content_parts.append(f"Profile URL: {profile_url}")

        page_content = "\n".join(page_content_parts)

        metadata = {
            "name":          name,
            "department":    department,
            "designation":   designation,
            "qualification": qualification,
            "awarded_year":  awarded_year,
            "date_of_join":  date_of_join,
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
            logger.warning(f"No faculty list found for {department}")
            return []

        docs: list[Document] = []
        seen_names: set[str] = set()

        for li in faculty_ul.find_all("li", recursive=False):
            doc = self._li_to_document(li, department)
            if doc is None:
                continue
            name = doc.metadata.get("name", "")
            if name in seen_names:
                continue
            seen_names.add(name)
            docs.append(doc)

        logger.info(f"{len(docs)} faculty extracted for [{department}]")
        return docs

    def scrape_all(self) -> list[Document]:
        all_docs: list[Document] = []

        for dept, url in FACULTY_URLS.items():
            logger.info(f"\n{'─' * 60}")
            logger.info(f"Dept : {dept}")
            logger.info(f"URL  : {url}")

            soup = self._get_soup(url)
            if soup is None:
                logger.warning(f"Skipping {dept} — page fetch failed.")
                continue

            dept_docs = self._scrape_page(soup, dept)
            all_docs.extend(dept_docs)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Total faculty documents: {len(all_docs)}")
        return all_docs

if __name__ == "__main__":
    import json

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")

    os.makedirs(DATA_DIR, exist_ok=True)

    scraper = FacultyScraper(delay=1.5)
    docs = scraper.scrape_all()

    output = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    with open(f"{DATA_DIR}/faculty_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(output)} faculty documents → faculty_data.json")
    for doc in output[:2]:
        print("\n--- Sample ---")
        print(doc["page_content"])
        print("Metadata:", doc["metadata"])