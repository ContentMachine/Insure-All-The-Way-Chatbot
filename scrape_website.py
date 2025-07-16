
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

# Starting URL
base_url = "https://insurealltheway.co"
visited_urls = set()
output_file = "insurealltheway_content.txt"

# Set up Selenium with ChromeDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
driver = webdriver.Chrome(options=options)

def clean_text(text):
    """Clean extracted text by removing excess whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text.strip())
    return text if text else "No content extracted"

def scrape_page(url):
    """Scrape text content from a single page using Selenium."""
    try:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator=' ')
        cleaned_text = clean_text(text)
        
        return cleaned_text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return f"Error: {str(e)}"

def get_internal_links(soup, base_url):
    """Find all internal links on a page."""
    internal_links = set()
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        parsed_url = urlparse(full_url)
        
        if parsed_url.netloc == base_domain:
            internal_links.add(full_url)
    
    return internal_links

def crawl_site(start_url):
    """Crawl the entire website and save content."""
    to_visit = {start_url}
    all_content = {}
    
    while to_visit:
        url = to_visit.pop()
        if url in visited_urls:
            continue
            
        print(f"Scraping: {url}")
        visited_urls.add(url)
        
        content = scrape_page(url)
        all_content[url] = content
        
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            internal_links = get_internal_links(soup, start_url)
            to_visit.update(internal_links - visited_urls)
        except Exception as e:
            print(f"Error crawling links from {url}: {e}")
            continue
        
        time.sleep(1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for url, content in all_content.items():
            f.write(f"--- Page: {url} ---\n")
            f.write(content + "\n\n")
    
    return all_content

if __name__ == "__main__":
    try:
        print(f"Starting crawl of {base_url}")
        crawl_site(base_url)
        print(f"Content saved to {output_file}")
    finally:
        driver.quit()