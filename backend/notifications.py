import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx
import logging

logger = logging.getLogger("rhythmai.notifications")

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ECG_QUERY = "ECG OR electrocardiogram[Title] AND (arrhythmia OR atrial fibrillation OR myocardial infarction)"


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


async def fetch_recent_ecg_articles(days: int = 3, max_results: int = 5) -> list[dict]:
    """Fetch recent ECG articles from PubMed."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
    articles: list[dict] = []

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            search_resp = await client.get(
                PUBMED_SEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": ECG_QUERY,
                    "mindate": since,
                    "datetype": "pdat",
                    "retmax": max_results,
                    "retmode": "json",
                },
            )
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            fetch_resp = await client.get(
                PUBMED_FETCH_URL,
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            )

            root = ET.fromstring(fetch_resp.text)
            for article in root.findall(".//PubmedArticle"):
                title = _text(article.find(".//ArticleTitle")) or "No title"
                pmid = _text(article.find(".//PMID"))
                abstract = _text(article.find(".//AbstractText"))
                pub_year = _text(article.find(".//PubDate/Year"))
                pub_date = pub_year or _text(article.find(".//ArticleDate/Year"))
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract[:300],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "https://pubmed.ncbi.nlm.nih.gov/",
                    "date": pub_date,
                })
    except Exception as exc:
        logger.exception("PubMed fetch error: %s", exc)

    return articles