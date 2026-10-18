import json
from pathlib import Path
from urllib.parse import urlparse

INCLUDE = {
    "claims-customer-service", "customer-service",
    "apartment-insurance", "car-insurance", "health-insurance",
    "travel-insurance", "life-insurance", "pets-insurance",
    "mortgage-insurance", "business-insurance", "other-insurances",
    "disability-and-incapacity-to-work",
    "pension-savings", "gemel", "savings", "ishtalmutfund", "loans",
}

# From robots.txt, we respect these
BLOCKED_PREFIXES = ("/more-information/", "/SPF", "/info/default.aspx",
                    "/MarketingCalculators", "/servicepages/sharedpages/discountsimulator")
BLOCKED_QUERY_KEYS = ("page=", "domain=", "tag=", "staff=")
SKIP_EXT = (".pdf", ".jpg", ".png", ".zip", ".xlsx", ".docx")

urls = Path("data/all_urls.txt").read_text(encoding="utf-8").splitlines()
selected = []
for u in urls:
    p = urlparse(u)
    parts = [x for x in p.path.split("/") if x]
    if not parts or parts[0] not in INCLUDE:
        continue
    if p.path.startswith(BLOCKED_PREFIXES):
        continue
    if any(k in p.query for k in BLOCKED_QUERY_KEYS):
        continue
    if p.path.lower().endswith(SKIP_EXT):
        continue
    selected.append(u)

Path("data/selected_urls.txt").write_text("\n".join(selected), encoding="utf-8")
print(f"Selected {len(selected)} of {len(urls)} URLs")