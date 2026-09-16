"""Small cached downloader: live reads fail explicitly; --offline replays saved inputs."""
import hashlib
import urllib.request
from datetime import datetime,timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = False

def cache_path(url):
    return ROOT / '.cache' / 'downloads' / (hashlib.sha256(url.encode()).hexdigest() + '.bin')

def provenance(url):
    path=cache_path(url)
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'fetched_at':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()}

def download(url):
    path = cache_path(url)
    if OFFLINE:
        return path.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent': 'Touchline educational forecasts'})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content
