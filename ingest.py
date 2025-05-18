import os
import re
import json
import time
import uuid
import pinecone
import openai
from typing import List, Dict, Any
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()
openai.api_key = OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_6F8B3i_Dxz2fopri1PV2RczmECDgQyE4GabRceXzVGrLMjx2CPVMzTYk76R5XYWEenB8Rr"
PINECONE_ENVIRONMENT = "us-east1-gcp"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "scraped-content"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_BATCH = 100

# --- Scraping Functions ---
def extract_links(article_text):
    link_pattern = r'(https?://[^\s]+|www\.[^\s]+\.[^\s]+)'
    links = re.finditer(link_pattern, article_text)
    links_dict = {}
    clean_text = article_text
    offset = 0
    for i, match in enumerate(links):
        link = match.group(0)
        start = match.start() - offset
        end = match.end() - offset
        links_dict[i] = {
            'link': link,
            'original_position': match.start(),
            'length': len(link)
        }
        clean_text = clean_text[:start] + clean_text[end:]
        offset += (end - start)
    return links_dict, clean_text

def setup_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up WebDriver: {e}")
        return None

def scrape_url_link(driver, url):
    if url.startswith('www.'):
        url = 'https://' + url
    result = {
        'url': url,
        'title': '',
        'text_content': '',
        'meta_description': '',
        'links': [],
        'status': 'failed',
        'error': None
    }
    try:
        driver.set_page_load_timeout(30)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)
        try:
            result['title'] = driver.title
        except:
            result['title'] = 'No title found'
        try:
            meta_desc = driver.find_element(By.CSS_SELECTOR, "meta[name='description']")
            result['meta_description'] = meta_desc.get_attribute("content")
        except:
            result['meta_description'] = 'No meta description found'
        try:
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            extracted_links = []
            for link in link_elements:
                href = link.get_attribute("href")
                text = link.text.strip()
                if href and href.startswith(('http://', 'https://', 'www.')):
                    extracted_links.append({
                        'url': href,
                        'text': text if text else 'No link text',
                        'title': link.get_attribute("title") or ''
                    })
            result['links'] = extracted_links
        except Exception as e:
            print(f"Error extracting links: {e}")
            result['links'] = []
        main_content = ''
        content_selectors = [
            "article", "main", ".content", "#content", ".post-content", ".article-content", ".entry-content", "#main-content"
        ]
        for selector in content_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    for elem in elements:
                        main_content += elem.text + "\n\n"
                    break
            except:
                continue
        if not main_content.strip():
            try:
                main_content = driver.find_element(By.TAG_NAME, "body").text
            except:
                main_content = 'Failed to extract content'
        result['text_content'] = main_content.strip()
        result['status'] = 'success'
        return result
    except TimeoutException:
        result['error'] = 'Page load timeout'
        return result
    except WebDriverException as e:
        result['error'] = f'WebDriver error: {str(e)}'
        return result
    except Exception as e:
        result['error'] = f'Unexpected error: {str(e)}'
        return result

def scrape_links_and_save(article_text, output_file='scraped_content.json'):
    links_dict, clean_text = extract_links(article_text)
    result = {
        'original_text': article_text,
        'clean_text': clean_text,
        'links': links_dict,
        'scraped_content': {}
    }
    driver = setup_webdriver()
    if not driver:
        print("Failed to set up WebDriver. Exiting.")
        return result
    try:
        for idx, link_info in links_dict.items():
            url = link_info['link']
            print(f"Scraping {url}...")
            scraped_data = scrape_url_link(driver, url)
            result['scraped_content'][idx] = scraped_data
            time.sleep(2)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    finally:
        if driver:
            driver.quit()
    return result

# --- Pinecone Ingestion Functions ---
def initialize_pinecone():
    from pinecone import ServerlessSpec
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-east1")
        )
        time.sleep(30)
    return pc.Index(PINECONE_INDEX_NAME)

def clean_and_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            if text[end-1] not in ".,!? ":
                last_period = text.rfind('.', start, end)
                last_space = text.rfind(' ', start, end)
                if last_period > start + (chunk_size // 2):
                    end = last_period + 1
                elif last_space > start + (chunk_size // 2):
                    end = last_space + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def get_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(chunks), MAX_CHUNKS_PER_BATCH):
        batch = chunks[i:i + MAX_CHUNKS_PER_BATCH]
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            if i + MAX_CHUNKS_PER_BATCH < len(chunks):
                time.sleep(0.5)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            embeddings.extend([[0] * EMBEDDING_DIMENSION] * len(batch))
    return embeddings

def ingest_json_file(file_path: str):
    print(f"Loading scraped content from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    index = initialize_pinecone()
    print("Processing original article...")
    article_text = data.get('original_text', '')
    article_chunks = clean_and_chunk_text(article_text)
    all_chunks = []
    all_metadata = []
    for i, chunk in enumerate(article_chunks):
        chunk_id = f"original_article_{i}"
        all_chunks.append(chunk)
        all_metadata.append({
            "id": chunk_id,
            "type": "original_article",
            "chunk_index": i,
            "total_chunks": len(article_chunks),
            "source": "original_article"
        })
    scraped_content = data.get('scraped_content', {})
    print(f"Processing {len(scraped_content)} scraped pages...")
    for page_id, page_data in scraped_content.items():
        if page_data.get('status') != 'success':
            continue
        url = page_data.get('url', '')
        title = page_data.get('title', '')
        text_content = page_data.get('text_content', '')
        if len(text_content) < 50:
            continue
        page_chunks = clean_and_chunk_text(text_content)
        for i, chunk in enumerate(page_chunks):
            chunk_id = f"page_{page_id}_chunk_{i}"
            all_chunks.append(chunk)
            all_metadata.append({
                "id": chunk_id,
                "type": "scraped_page",
                "url": url,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(page_chunks),
                "page_id": page_id
            })
            if i == 0 and page_data.get('links'):
                link_texts = []
                for link in page_data.get('links', []):
                    link_url = link.get('url', '')
                    link_text = link.get('text', '')
                    if link_url and link_text:
                        link_texts.append(f"{link_text}: {link_url}")
                if link_texts:
                    links_text = "Links found on page:\n" + "\n".join(link_texts)
                    links_chunks = clean_and_chunk_text(links_text)
                    for j, link_chunk in enumerate(links_chunks):
                        chunk_id = f"page_{page_id}_links_{j}"
                        all_chunks.append(link_chunk)
                        all_metadata.append({
                            "id": chunk_id,
                            "type": "page_links",
                            "url": url,
                            "title": title + " - Links",
                            "chunk_index": j,
                            "total_chunks": len(links_chunks),
                            "page_id": page_id
                        })
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = get_embeddings(all_chunks)
    vectors = []
    for i, (chunk, embedding, metadata) in enumerate(zip(all_chunks, embeddings, all_metadata)):
        vector_id = str(uuid.uuid4())
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                **metadata,
                "text": chunk[:1000],
                "timestamp": time.time()
            }
        })
        if len(vectors) >= 100 or i == len(all_chunks) - 1:
            print(f"Upserting batch of {len(vectors)} vectors to Pinecone...")
            index.upsert(vectors=vectors)
            vectors = []
    print("Ingestion complete!")
    return len(all_chunks)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape and ingest article and links into Pinecone')
    parser.add_argument('--article', help='Path to text file containing article, or pass the article as a string')
    parser.add_argument('--json', help='If you already have a scraped_content.json, ingest it directly')
    args = parser.parse_args()
    if args.article:
        if os.path.isfile(args.article):
            with open(args.article, 'r', encoding='utf-8') as f:
                article_text = f.read()
        else:
            article_text = args.article
        scrape_links_and_save(article_text, 'scraped_content.json')
        ingest_json_file('scraped_content.json')
    elif args.json:
        ingest_json_file(args.json)
    else:
        print('Please provide --article <file_or_text> or --json <scraped_content.json>') 