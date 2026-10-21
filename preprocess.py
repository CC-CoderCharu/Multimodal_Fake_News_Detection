

#!/usr/bin/env python3
"""Preprocess locally available multimodal fake-news data and collect missing media from URLs."""

from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
import threading
import urllib.parse
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
META = ROOT / "metadata"
MASTER = META / "master_dataset.csv"
REPORT = META / "preprocessing_report.txt"

# Concurrency & Limit Constants
MAX_REDDIT_SAMPLES = 10000
MAX_FNN_SAMPLES = 1000
NUM_THREADS = 25

IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile(r"\s+")

STATS = defaultdict(int)
STATS_LOCK = threading.Lock()

# Thread-safe counter helpers
def increment_stat(key, amount=1):
    with STATS_LOCK:
        STATS[key] += amount

def get_stat(key):
    with STATS_LOCK:
        return STATS[key]


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = html.unescape(text)
    text = HTML_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = CTRL_RE.sub(" ", text)
    text = URL_RE.sub("[URL]", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def download_and_verify_image(url: str, save_path: Path) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=4)
        if r.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(r.content)
            # Verify using Pillow
            with Image.open(save_path) as img:
                img.verify()
            return True
    except Exception:
        pass
    
    # Cleanup on failure
    if save_path.exists():
        try:
            save_path.unlink()
        except Exception:
            pass
    return False


def scrape_article(url: str) -> tuple[str | None, str | None]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        # Import BeautifulSoup locally inside function
        from bs4 import BeautifulSoup
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        r = requests.get(url, headers=headers, timeout=4)
        if r.status_code != 200:
            return None, None
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Extract body text
        p_texts = [p.get_text().strip() for p in soup.find_all("p")]
        body_text = " ".join([t for t in p_texts if len(t) > 10])
        
        # Extract image URL
        img_url = None
        # Check og:image meta tag
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            img_url = og_img.get("content")
        else:
            # Check twitter:image meta tag
            tw_img = soup.find("meta", name="twitter:image")
            if tw_img and tw_img.get("content"):
                img_url = tw_img.get("content")
            else:
                # Fallback: check link rel="image_src"
                link_img = soup.find("link", rel="image_src")
                if link_img and link_img.get("href"):
                    img_url = link_img.get("href")
                else:
                    # Fallback: first absolute image src
                    for im in soup.find_all("img"):
                        src = im.get("src")
                        if src and src.startswith("http"):
                            img_url = src
                            break
        if img_url:
            img_url = urllib.parse.urljoin(url, img_url)
        return body_text, img_url
    except Exception:
        return None, None


def process_image(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im = im.resize((224, 224), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=90)
        return True
    except Exception:
        increment_stat("corrupted_images")
        return False


def map_label(val, source: str = "reddit") -> int | None:
    if pd.isna(val):
        return None
    try:
        v = int(val)
    except (TypeError, ValueError):
        s = str(val).strip().lower()
        if s in {"0", "real", "true", "non-rumour", "non-rumor"}:
            return 0
        if s in {"1", "fake", "false", "rumour", "rumor"}:
            return 1
        return None
    if source == "reddit":
        if v in (0, 1, 2, 3, 4, 5):
            return v
    else:
        if v in (0, 1):
            return v
    return None


def add_candidate(cands: list, source: str, original_id: str, label, title: str, text: str, image: Path | None):
    increment_stat(f"raw_{source}")
    oid = str(original_id).strip()
    if not oid or oid.lower() in {"nan", "none"}:
        increment_stat("invalid_id")
        return
    lab = map_label(label, source=source)
    if lab is None:
        increment_stat("invalid_label")
        return
    title_c = clean_text(title)
    text_c = clean_text(text) or title_c
    if not text_c:
        increment_stat("missing_text")
        return
        
    if source != "twitter":
        if image is None or not image.is_file():
            increment_stat("missing_images")
            return
            
    cands.append(
        {
            "source": source,
            "original_id": oid,
            "label": lab,
            "title": title_c,
            "text": text_c,
            "raw_image": image,
        }
    )


def reddit_download_task(oid, url, title, text, label, raw_img_path, limit, counter_fn):
    if counter_fn() >= limit:
        return None
    success = download_and_verify_image(url, raw_img_path)
    if success:
        return (oid, label, title, text, raw_img_path)
    else:
        increment_stat("download_failed")
    return None


def load_existing_master() -> tuple[list[dict], set[tuple[str, str]], set[str], set[str], set[tuple[str, str]], dict[str, int]]:
    existing_rows = []
    seen_ids = set()
    seen_text = set()
    seen_img = set()
    seen_pair = set()
    max_indices = {"reddit": 0, "fakenewsnet": 0, "twitter": 0}
    
    if not MASTER.exists():
        return existing_rows, seen_ids, seen_text, seen_img, seen_pair, max_indices
        
    try:
        df = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
        existing_rows = df.to_dict("records")
        for r in existing_rows:
            r["label"] = int(r["label"])
            source = r["source"]
            orig_id = r["original_id"]
            seen_ids.add((source, orig_id))
            
            # Extract index from sample_id e.g. REDDIT_000012 -> 12
            sid = r["sample_id"]
            try:
                idx_str = sid.split("_")[-1]
                idx = int(idx_str)
                if idx > max_indices[source]:
                    max_indices[source] = idx
            except Exception:
                pass
            
            # Read clean text file if exists to get hash
            tpath = ROOT / r["text_path"]
            th = None
            if tpath.is_file():
                try:
                    text_content = tpath.read_text(encoding="utf-8")
                    th = md5(text_content.encode("utf-8")).hexdigest()
                    seen_text.add(th)
                except Exception:
                    pass
                    
            # Check raw image bytes hash to get image hash
            ih = None
            if source != "twitter":
                # First check raw image
                raw_img_path = None
                if source == "reddit":
                    raw_img_path = RAW / "reddit" / "images" / f"{orig_id}.jpg"
                elif source == "fakenewsnet":
                    raw_img_path = RAW / "fakenewsnet" / "images" / f"{orig_id}.jpg"
                    
                if raw_img_path and raw_img_path.is_file():
                    try:
                        ih = md5(raw_img_path.read_bytes()).hexdigest()
                        seen_img.add(ih)
                    except Exception:
                        pass
                else:
                    # Fallback to processed image if raw image is deleted
                    ipath = ROOT / r["image_path"]
                    if ipath.is_file():
                        try:
                            ih = md5(ipath.read_bytes()).hexdigest()
                            seen_img.add(ih)
                        except Exception:
                            pass
            if th and ih:
                seen_pair.add((th, ih))
                
        print(f"Loaded {len(existing_rows)} existing records from master dataset.")
        print(f"Max indices: {max_indices}")
    except Exception as e:
        print(f"Error loading existing master: {e}")
        
    return existing_rows, seen_ids, seen_text, seen_img, seen_pair, max_indices


def load_baseline_stats():
    if not REPORT.exists():
        return
    try:
        content = REPORT.read_text(encoding="utf-8")
        
        def find_count(pattern, text):
            match = re.search(pattern, text)
            return int(match.group(1)) if match else 0
            
        STATS["raw_reddit"] = find_count(r"-\s+Reddit:\s+(\d+)", content)
        STATS["raw_twitter"] = find_count(r"-\s+PHEME/Twitter:\s+(\d+)", content)
        STATS["raw_fakenewsnet"] = find_count(r"-\s+FakeNewsNet:\s+(\d+)", content)
        
        STATS["missing_text"] = find_count(r"-\s+Missing text:\s+(\d+)", content)
        STATS["download_failed"] = find_count(r"-\s+Missing/failed images:\s+(\d+)", content)
        STATS["corrupted_images"] = find_count(r"-\s+Corrupted images:\s+(\d+)", content)
        STATS["invalid_label"] = find_count(r"-\s+Invalid labels:\s+(\d+)", content)
        
        STATS["dup_original_id"] = find_count(r"\(\s*original_id=(\d+)", content)
        STATS["dup_text"] = find_count(r"text=(\d+)", content)
        STATS["dup_image"] = find_count(r"image=(\d+)", content)
        STATS["dup_pair"] = find_count(r"pair=(\d+)", content)
        STATS["duplicates_removed"] = find_count(r"-\s+Duplicates removed:\s+(\d+)", content)
        
        print(f"Loaded baseline stats from report: {dict(STATS)}")
    except Exception as e:
        print(f"Error loading baseline stats: {e}")


def load_reddit(cands: list, completed_ids: set[tuple[str, str]], existing_count: int):
    if existing_count >= MAX_REDDIT_SAMPLES:
        print(f"Reddit: Already have {existing_count} samples, which meets target {MAX_REDDIT_SAMPLES}. Skipping.")
        return

    reddit_dir = RAW / "reddit"
    files = [
        reddit_dir / "multimodal_test_public.tsv",
        reddit_dir / "multimodal_validate.tsv",
        reddit_dir / "multimodal_train.tsv",
    ]
    
    tasks = []
    for f in files:
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        print(f"Loaded {f.name}: {len(df)} rows")
        cols = {c.lower(): c for c in df.columns}
        id_col = cols.get("id")
        lab_col = cols.get("6_way_label") or cols.get("2_way_label") or cols.get("label")
        title_col = cols.get("title")
        text_col = cols.get("clean_title") or cols.get("title")
        url_col = cols.get("image_url")
        
        for row in df.itertuples(index=False, name=None):
            rec = dict(zip(df.columns, row))
            oid = str(rec.get(id_col, ""))
            url = str(rec.get(url_col, ""))
            label = rec.get(lab_col)
            title = rec.get(title_col, "")
            text = rec.get(text_col, "")
            
            if oid and url and label is not None:
                if ("reddit", oid) in completed_ids:
                    continue
                tasks.append((oid, url, title, text, label))
                
    reddit_img_dir = reddit_dir / "images"
    reddit_img_dir.mkdir(parents=True, exist_ok=True)
    
    cached_cands = []
    pending_tasks = []
    successful_count = existing_count
    
    for oid, url, title, text, label in tasks:
        raw_img_path = reddit_img_dir / f"{oid}.jpg"
        if raw_img_path.exists():
            cached_cands.append((oid, label, title, text, raw_img_path))
            successful_count += 1
            if successful_count >= MAX_REDDIT_SAMPLES:
                break
        else:
            if len(pending_tasks) + successful_count < MAX_REDDIT_SAMPLES + 1000:
                pending_tasks.append((oid, url, title, text, label, raw_img_path))
                
    print(f"Reddit: {len(cached_cands) - (successful_count - existing_count)} cached images found, total successes: {successful_count}/{MAX_REDDIT_SAMPLES}")
    
    # Download missing images concurrently
    if pending_tasks and successful_count < MAX_REDDIT_SAMPLES:
        # We allow target to be slightly higher (e.g. +200) to compensate for duplicate removal losses
        download_target = MAX_REDDIT_SAMPLES + 200
        print(f"Downloading Fakeddit images ({len(pending_tasks)} queued, target successes: {download_target})...")
        
        successful_downloads = successful_count
        dl_lock = threading.Lock()
        
        def get_current_success():
            with dl_lock:
                return successful_downloads
                
        def increment_current_success():
            nonlocal successful_downloads
            with dl_lock:
                successful_downloads += 1
                return successful_downloads
        
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = []
            for oid, url, title, text, label, raw_img_path in pending_tasks:
                futures.append(
                    executor.submit(
                        reddit_download_task, 
                        oid, url, title, text, label, raw_img_path, 
                        download_target, get_current_success
                    )
                )
                
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    oid, label, title, text, raw_img_path = res
                    cached_cands.append((oid, label, title, text, raw_img_path))
                    curr = increment_current_success()
                    if curr % 200 == 0 or curr >= download_target:
                        print(f"  Fakeddit downloads: {curr}/{download_target}")
                    if curr >= download_target:
                        break
                        
    # Register as candidates
    for oid, label, title, text, raw_img_path in cached_cands:
        add_candidate(cands, "reddit", oid, label, title, text, raw_img_path)


def load_pheme(cands: list):
    pheme = RAW / "twitter" / "pheme-rnr-dataset"
    if not pheme.exists():
        return
    for jf in pheme.rglob("source-tweet"):
        for path in jf.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            oid = str(data.get("id_str") or data.get("id") or path.stem)
            text = data.get("full_text") or data.get("text") or ""
            parts = {p.lower() for p in path.parts}
            label = 0 if "non-rumours" in parts or "non-rumors" in parts else 1
            add_candidate(cands, "twitter", oid, label, "", text, None)


def fnn_scrape_task(news_id, url, title, label, limit, counter_fn):
    if counter_fn() >= limit:
        return None
        
    cache_path = RAW / "fakenewsnet" / "raw_articles" / f"{news_id}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already cached (failure or success)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("success"):
                img_path = Path(data.get("image_path"))
                if img_path.exists():
                    return (news_id, label, title, data.get("body_text", ""), img_path)
            else:
                return None
        except Exception:
            pass
            
    # Crawl the article
    body_text, img_url = scrape_article(url)
    if not body_text or not img_url:
        # Cache failure
        try:
            cache_path.write_text(json.dumps({
                "id": news_id,
                "url": url,
                "title": title,
                "success": False
            }, indent=2), encoding="utf-8")
        except Exception:
            pass
        return None
        
    raw_img_path = RAW / "fakenewsnet" / "images" / f"{news_id}.jpg"
    img_success = download_and_verify_image(img_url, raw_img_path)
    if not img_success:
        # Cache failure
        try:
            cache_path.write_text(json.dumps({
                "id": news_id,
                "url": url,
                "title": title,
                "success": False
            }, indent=2), encoding="utf-8")
        except Exception:
            pass
        return None
        
    # Cache success
    try:
        cache_path.write_text(json.dumps({
            "id": news_id,
            "url": url,
            "title": title,
            "body_text": body_text,
            "image_path": str(raw_img_path),
            "success": True
        }, indent=2), encoding="utf-8")
    except Exception:
        pass
        
    return (news_id, label, title, body_text, raw_img_path)


def load_fnn(cands: list, completed_ids: set[tuple[str, str]], existing_count: int):
    # Clean up any failed Gossipcop cache files that were caused by the missing URL schema
    raw_articles_dir = RAW / "fakenewsnet" / "raw_articles"
    if raw_articles_dir.exists():
        deleted_failures = 0
        for f in raw_articles_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not data.get("success") and not data.get("url", "").startswith(("http://", "https://")):
                    f.unlink()
                    deleted_failures += 1
            except Exception:
                pass
        if deleted_failures > 0:
            print(f"Cleaned up {deleted_failures} failed Gossipcop caches that lacked URL schemas to allow retry.")

    if existing_count >= MAX_FNN_SAMPLES:
        print(f"FakeNewsNet: Already have {existing_count} samples, which meets target {MAX_FNN_SAMPLES}. Skipping.")
        return

    ds = RAW / "fakenewsnet" / "dataset"
    mapping = {
        "politifact_fake.csv": 1,
        "politifact_real.csv": 0,
        "gossipcop_fake.csv": 1,
        "gossipcop_real.csv": 0,
    }
    
    tasks = []
    for name, label in mapping.items():
        f = ds / name
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        print(f"Loaded {name}: {len(df)} rows")
        for rec in df.to_dict("records"):
            news_id = rec.get("id", "")
            news_url = rec.get("news_url", "")
            title = rec.get("title", "")
            if news_id and news_url:
                if ("fakenewsnet", news_id) in completed_ids:
                    continue
                tasks.append((news_id, news_url, title, label))
                
    # Create scraping cache directories
    (RAW / "fakenewsnet" / "images").mkdir(parents=True, exist_ok=True)
    (RAW / "fakenewsnet" / "raw_articles").mkdir(parents=True, exist_ok=True)
    
    # Separate existing cached vs new ones to be processed
    cached_cands = []
    pending_tasks = []
    successful_count = existing_count
    
    for news_id, news_url, title, label in tasks:
        cache_path = RAW / "fakenewsnet" / "raw_articles" / f"{news_id}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if data.get("success"):
                    img_path = Path(data.get("image_path"))
                    if img_path.exists():
                        cached_cands.append((news_id, label, title, data.get("body_text", ""), img_path))
                        successful_count += 1
                        if successful_count >= MAX_FNN_SAMPLES:
                            break
            except Exception:
                pass
        else:
            if len(pending_tasks) + successful_count < MAX_FNN_SAMPLES + 500:
                pending_tasks.append((news_id, news_url, title, label))
                
    print(f"FakeNewsNet: {len(cached_cands) - (successful_count - existing_count)} cached articles found, total successes: {successful_count}/{MAX_FNN_SAMPLES}")
    
    # Scrape missing ones concurrently
    if pending_tasks and successful_count < MAX_FNN_SAMPLES:
        # Allow target to be slightly higher to compensate for duplicate removal losses
        scrape_target = MAX_FNN_SAMPLES + 50
        print(f"Scraping/Crawling FakeNewsNet articles ({len(pending_tasks)} queued, target successes: {scrape_target})...")
        
        successful_downloads = successful_count
        dl_lock = threading.Lock()
        
        def get_current_success():
            with dl_lock:
                return successful_downloads
                
        def increment_current_success():
            nonlocal successful_downloads
            with dl_lock:
                successful_downloads += 1
                return successful_downloads
                
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = []
            for news_id, news_url, title, label in pending_tasks:
                futures.append(
                    executor.submit(
                        fnn_scrape_task,
                        news_id, news_url, title, label,
                        scrape_target, get_current_success
                    )
                )
                
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    news_id, label, title, body_text, raw_img_path = res
                    cached_cands.append((news_id, label, title, body_text, raw_img_path))
                    curr = increment_current_success()
                    if curr % 50 == 0 or curr >= scrape_target:
                        print(f"  FakeNewsNet crawls: {curr}/{scrape_target}")
                    if curr >= scrape_target:
                        break
                        
    # Register as candidates
    for news_id, label, title, body_text, raw_img_path in cached_cands:
        add_candidate(cands, "fakenewsnet", news_id, label, title, body_text, raw_img_path)


def dedup_new(cands: list, seen_text: set[str], seen_img: set[str], seen_pair: set[tuple[str, str]]) -> list:
    out = []
    
    for c in cands:
        th = md5(c["text"].encode("utf-8")).hexdigest()
        
        if c["raw_image"]:
            try:
                ih = md5(c["raw_image"].read_bytes()).hexdigest()
            except Exception:
                increment_stat("corrupted_images")
                continue
        else:
            ih = None
            
        if th in seen_text:
            increment_stat("dup_text")
            continue
            
        if ih is not None:
            if ih in seen_img:
                increment_stat("dup_image")
                continue
            pair = (th, ih)
            if pair in seen_pair:
                increment_stat("dup_pair")
                continue
            seen_img.add(ih)
            seen_pair.add(pair)
            
        seen_text.add(th)
        out.append(c)
        
    with STATS_LOCK:
        STATS["duplicates_removed"] = (
            STATS["dup_original_id"] + STATS["dup_text"] + STATS["dup_image"] + STATS["dup_pair"]
        )
    return out


def assign_and_write_new(cands: list, max_indices: dict[str, int]) -> list[dict]:
    # Reset and prepare directories (do not clear them, just ensure they exist)
    for src_type in ("reddit", "fakenewsnet", "twitter"):
        (PROC / src_type / "text").mkdir(parents=True, exist_ok=True)
        if src_type != "twitter":
            (PROC / src_type / "images").mkdir(parents=True, exist_ok=True)
            
    counters = dict(max_indices)
    prefixes = {"reddit": "REDDIT", "twitter": "TWITTER", "fakenewsnet": "FNN"}
    rows = []
    
    for c in cands:
        src = c["source"]
        counters[src] += 1
        sid = f"{prefixes[src]}_{counters[src]:06d}"
        
        tpath = PROC / src / "text" / f"{sid}.txt"
        
        if src != "twitter":
            ipath = PROC / src / "images" / f"{sid}.jpg"
            if not process_image(c["raw_image"], ipath):
                counters[src] -= 1
                increment_stat("failed_images")
                continue
            tpath.write_text(c["text"], encoding="utf-8")
            rows.append(
                {
                    "sample_id": sid,
                    "source": src,
                    "original_id": c["original_id"],
                    "label": c["label"],
                    "title": c["title"],
                    "text_path": f"data/processed/{src}/text/{sid}.txt",
                    "image_path": f"data/processed/{src}/images/{sid}.jpg",
                }
            )
        else:
            tpath.write_text(c["text"], encoding="utf-8")
            rows.append(
                {
                    "sample_id": sid,
                    "source": src,
                    "original_id": c["original_id"],
                    "label": c["label"],
                    "title": c["title"],
                    "text_path": f"data/processed/{src}/text/{sid}.txt",
                    "image_path": "",
                }
            )
            
    return rows


def quality_check(rows: list[dict]) -> list[str]:
    issues = []
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        issues.append("duplicate sample_id")
        
    pairs = set()
    for r in rows:
        tp = ROOT / r["text_path"]
        if not tp.is_file():
            issues.append(f"missing text {r['sample_id']}")
            
        if r["source"] != "twitter":
            ip = ROOT / r["image_path"]
            if not ip.is_file():
                issues.append(f"missing image {r['sample_id']}")
                continue
                
            tid = tp.stem
            iid = ip.stem
            if tid != r["sample_id"] or iid != r["sample_id"] or tid != iid:
                issues.append(f"id mismatch {r['sample_id']}")
                
            valid_labels = (0, 1, 2, 3, 4, 5) if r["source"] == "reddit" else (0, 1)
            if int(r["label"]) not in valid_labels:
                issues.append(f"bad label {r['sample_id']}")
                
            try:
                with Image.open(ip) as im:
                    if im.mode != "RGB" or im.size != (224, 224):
                        issues.append(f"bad image shape {r['sample_id']}")
            except Exception:
                issues.append(f"unreadable {r['sample_id']}")
                
            pair = (tp.read_text(encoding="utf-8"), ip.stat().st_size)
            if pair in pairs:
                issues.append(f"dup pair {r['sample_id']}")
            pairs.add(pair)
        else:
            if r.get("image_path") and not pd.isna(r["image_path"]) and str(r["image_path"]).strip() != "":
                issues.append(f"unexpected image for twitter {r['sample_id']}")
            valid_labels = (0, 1, 2, 3, 4, 5) if r["source"] == "reddit" else (0, 1)
            if int(r["label"]) not in valid_labels:
                issues.append(f"bad label {r['sample_id']}")
                
    return issues


def write_outputs(rows: list[dict], issues: list[str]):
    META.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "source", "original_id", "label", "title", "text_path", "image_path"]
    with MASTER.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
    fake = sum(1 for r in rows if r["label"] == 1)
    real = sum(1 for r in rows if r["label"] == 0)
    by = defaultdict(int)
    for r in rows:
        by[r["source"]] += 1
        
    raw_total = STATS["raw_reddit"] + STATS["raw_twitter"] + STATS["raw_fakenewsnet"]
    
    # Calculate processed paths count
    proc_reddit_text = len(list((PROC / "reddit" / "text").glob("*.txt"))) if (PROC / "reddit" / "text").exists() else 0
    proc_reddit_img = len(list((PROC / "reddit" / "images").glob("*.jpg"))) if (PROC / "reddit" / "images").exists() else 0
    proc_fnn_text = len(list((PROC / "fakenewsnet" / "text").glob("*.txt"))) if (PROC / "fakenewsnet" / "text").exists() else 0
    proc_fnn_img = len(list((PROC / "fakenewsnet" / "images").glob("*.jpg"))) if (PROC / "fakenewsnet" / "images").exists() else 0
    proc_twitter_text = len(list((PROC / "twitter" / "text").glob("*.txt"))) if (PROC / "twitter" / "text").exists() else 0
    
    report = f"""PREPROCESSING REPORT
====================
Raw locations (read-only):
- Reddit/Fakeddit: data/raw/reddit/ (multimodal_train/validate/test_public.tsv)
- PHEME: data/raw/twitter/pheme-rnr-dataset/
- FakeNewsNet: data/raw/fakenewsnet/dataset/*.csv

RAW SAMPLE COUNTS
- Reddit: {STATS['raw_reddit']}
- PHEME/Twitter: {STATS['raw_twitter']}
- FakeNewsNet: {STATS['raw_fakenewsnet']}
- Raw total: {raw_total}

FINAL PROCESSED SAMPLES: {len(rows)}
- Fake (1): {fake}
- Real (0): {real}
- Reddit pairs: {by['reddit']}
- FakeNewsNet pairs: {by['fakenewsnet']}
- Twitter/PHEME text samples: {by['twitter']}
- Processed Reddit text: {proc_reddit_text}
- Processed Reddit images: {proc_reddit_img}
- Processed FakeNewsNet text: {proc_fnn_text}
- Processed FakeNewsNet images: {proc_fnn_img}
- Processed Twitter text: {proc_twitter_text}

REMOVED / EXCLUDED / FAILED
- Missing text: {STATS['missing_text']}
- Missing/failed images: {STATS['missing_images'] + STATS['failed_images'] + STATS['download_failed']}
- Corrupted images: {STATS['corrupted_images']}
- Invalid labels: {STATS['invalid_label']}
- Duplicates removed: {STATS['duplicates_removed']}
  (original_id={STATS['dup_original_id']}, text={STATS['dup_text']}, image={STATS['dup_image']}, pair={STATS['dup_pair']})

PREPROCESSING
- Text: HTML unescape, strip tags, Unicode NFKC, whitespace/control cleanup, URLs -> [URL]. Preserved punctuation/hashtags/mentions. No stemming/lemmatizing/stopword/BERT.
- Images: EXIF transpose, RGB, resize 224x224, saved as JPEG. Raw files are not modified.

FILES CREATED
- metadata/master_dataset.csv
- metadata/preprocessing_report.txt
- data/processed/<source>/text/<sample_id>.txt
- data/processed/<source>/images/<sample_id>.jpg

QUALITY CHECK
- Issues detected: {len(issues)}
{chr(10).join(['  - ' + iss for iss in issues[:20]])}
"""
    REPORT.write_text(report, encoding="utf-8")


def main():
    print("Starting Data Collection and Preprocessing Pipeline...")
    
    # Load baseline statistics and completed master entries
    load_baseline_stats()
    existing_rows, completed_ids, seen_text, seen_img, seen_pair, max_indices = load_existing_master()
    
    counts = {
        "reddit": sum(1 for r in existing_rows if r["source"] == "reddit"),
        "fakenewsnet": sum(1 for r in existing_rows if r["source"] == "fakenewsnet"),
        "twitter": sum(1 for r in existing_rows if r["source"] == "twitter")
    }
    
    cands: list = []
    
    print("\n[1/3] Reading & Crawling FakeNewsNet...")
    load_fnn(cands, completed_ids, counts["fakenewsnet"])
    
    print("\n[2/3] Reading & Downloading Fakeddit (Reddit)...")
    load_reddit(cands, completed_ids, counts["reddit"])
    
    print("\n[3/3] Reading Twitter/PHEME...")
    if counts["twitter"] == 0:
        load_pheme(cands)
    else:
        print(f"Twitter: Already have {counts['twitter']} samples. Skipping reading raw Twitter files.")
    
    print(f"\nTotal new candidates loaded before deduplication: {len(cands)}")
    
    print("\nRunning Deduplication...")
    new_cands = dedup_new(cands, seen_text, seen_img, seen_pair)
    print(f"New candidates remaining after deduplication: {len(new_cands)}")
    
    print("\nProcessing and Saving Dataset...")
    new_rows = assign_and_write_new(new_cands, max_indices)
    
    # Combine existing rows with newly processed ones
    combined_rows = existing_rows + new_rows
    
    print("\nRunning Quality Checks...")
    issues = quality_check(combined_rows)
    if issues:
        print(f"  WARNING: Quality issues detected: {len(issues)}")
        
    write_outputs(combined_rows, issues)
    
    # Calculate required display stats
    reddit_pairs = sum(1 for r in combined_rows if r["source"] == "reddit")
    fnn_pairs = sum(1 for r in combined_rows if r["source"] == "fakenewsnet")
    twitter_samples = sum(1 for r in combined_rows if r["source"] == "twitter")
    total_images = reddit_pairs + fnn_pairs
    fake_count = sum(1 for r in combined_rows if r["label"] == 1)
    real_count = sum(1 for r in combined_rows if r["label"] == 0)
    duplicates_removed = STATS["duplicates_removed"]
    failed_images = STATS["failed_images"] + STATS["corrupted_images"] + STATS["download_failed"]
    
    print("\n==================================================")
    print("FINAL STATS OUTPUT")
    print("==================================================")
    print(f"Reddit text-image pairs: {reddit_pairs}")
    print(f"FakeNewsNet text-image pairs: {fnn_pairs}")
    print(f"Twitter/PHEME text samples: {twitter_samples}")
    print(f"Total images: {total_images}")
    print(f"Fake: {fake_count}")
    print(f"Real: {real_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Failed images: {failed_images}")
    print()
    print("metadata/master_dataset.csv")
    print("metadata/preprocessing_report.txt")
    print("==================================================")


if __name__ == "__main__":
    main()
