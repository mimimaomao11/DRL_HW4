"""Tool 1: ArXiv Paper Search — queries ArXiv API and returns structured paper metadata."""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


def search_arxiv(query: str, max_results: int = 5) -> dict:
    """Search ArXiv for research papers matching the query.

    Args:
        query: Search string (e.g. "PPO reinforcement learning robot")
        max_results: Maximum number of papers to return (default 5)

    Returns:
        dict with keys: papers (list), count (int), query (str)
    """
    base_url = "http://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })

    try:
        with urllib.request.urlopen(base_url + params, timeout=10) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        return {"error": f"ArXiv API request failed: {e}", "papers": [], "count": 0}

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        link = entry.find("atom:id", ns)
        authors = entry.findall("atom:author", ns)

        papers.append({
            "title": title.text.strip().replace("\n", " ") if title is not None else "N/A",
            "authors": [
                a.find("atom:name", ns).text
                for a in authors[:3]
                if a.find("atom:name", ns) is not None
            ],
            "abstract": (summary.text.strip()[:300] + "...") if summary is not None else "N/A",
            "url": link.text.strip() if link is not None else "N/A",
            "published": published.text[:10] if published is not None else "N/A",
        })

    return {"papers": papers, "count": len(papers), "query": query}
