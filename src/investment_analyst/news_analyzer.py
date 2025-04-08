import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Any, Optional, Set
import logging
import schedule
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from ..utils.common import setup_logging, save_to_json, get_current_time
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from functools import wraps
import feedparser
import hashlib
from urllib.parse import urlparse
import yfinance as yf
import re

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')

class NewsAnalyzer:
    def __init__(self):
        self.logger = setup_logging('news_analyzer')
        self.sia = SentimentIntensityAnalyzer()
        self.market_hours = {
            'start': '09:30',
            'end': '16:00'
        }
        
        # List of user agents to rotate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
        ]
        
        # News sources to scrape with updated URLs
        self.news_sources = {
            'yahoo_finance': 'https://query2.finance.yahoo.com/v1/finance/search?q=market&quotesCount=0&newsCount=20&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&newsQueryId=news_cie_vespa&enableCb=true&enableNavLinks=true&enableEnhancedTrivialQuery=true',
            'cnbc': 'https://www.cnbc.com/world/?region=world',
            'marketwatch': 'https://www.marketwatch.com/latest-news',
            'cnn_business': 'https://www.cnn.com/business/markets',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'ft': 'https://www.ft.com/markets?format=rss'
        }
        
        # Initialize empty proxy list
        self.proxies = []
        
        # Setup session with retry strategy
        self.session = self._setup_session()
        
        # Initialize company and sector keywords
        self.company_keywords = self._load_company_keywords()
        self.sector_keywords = self._load_sector_keywords()
        
        # Cache for article deduplication
        self.article_cache = set()

    def scrape_yahoo_finance(self) -> List[Dict[str, Any]]:
        """Scrape news from Yahoo Finance using their API"""
        articles = []
        try:
            # First try the API endpoint
            response = self.safe_request(self.news_sources['yahoo_finance'])
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    if 'news' in data:
                        for item in data['news']:
                            try:
                                article = {
                                    'title': item.get('title', ''),
                                    'url': item.get('link', ''),
                                    'source': 'Yahoo Finance',
                                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', time.time())).isoformat()
                                }
                                
                                if article['title'] and article['url'] and not self.is_duplicate(article):
                                    articles.append(article)
                            except Exception as e:
                                self.logger.error(f"Error processing Yahoo Finance API article: {str(e)}")
                                continue
                except Exception as e:
                    self.logger.error(f"Error parsing Yahoo Finance API response: {str(e)}")
            
            # If API fails or returns no results, try web scraping
            if not articles:
                web_url = 'https://finance.yahoo.com/topic/stock-market-news'
                response = self.safe_request(web_url)
                if response and response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Try multiple selectors for news items
                    news_items = (
                        soup.find_all('h3', {'class': ['Mb(5px)']}) or
                        soup.find_all('div', {'data-test': 'story'}) or
                        soup.find_all('div', {'class': ['Pos(r)']}) or
                        soup.find_all('div', {'class': ['js-stream-content']})
                    )
                    
                    for item in news_items:
                        try:
                            # Find title and link
                            title_elem = (
                                item.find('a', {'data-test': 'story-title'}) or
                                item.find('a', {'class': ['js-content-viewer']}) or
                                item.find('a')
                            )
                            
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                url = title_elem.get('href', '')
                                
                                # Handle relative URLs
                                if url.startswith('/'):
                                    url = 'https://finance.yahoo.com' + url
                                elif not url.startswith('http'):
                                    url = 'https://finance.yahoo.com/' + url
                                
                                article = {
                                    'title': title,
                                    'url': url,
                                    'source': 'Yahoo Finance',
                                    'published_at': datetime.now().isoformat()
                                }
                                
                                if not self.is_duplicate(article):
                                    articles.append(article)
                                    
                        except Exception as e:
                            self.logger.error(f"Error processing Yahoo Finance article: {str(e)}")
                            continue
                            
        except Exception as e:
            self.logger.error(f"Error scraping Yahoo Finance: {str(e)}")
            
        return articles

    def scrape_cnbc(self) -> List[Dict[str, Any]]:
        """Scrape news from CNBC"""
        articles = []
        try:
            response = self.safe_request(self.news_sources['cnbc'])
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                news_items = soup.find_all('div', {'class': 'Card-standardBreakerCard'}) or \
                            soup.find_all('div', {'class': 'Card-title'}) or \
                            soup.find_all('div', {'class': 'Card-headline'})
                
                for item in news_items:
                    title_elem = item.find('a', {'class': 'Card-title'}) or item.find('a')
                    if title_elem:
                        title = title_elem.text.strip()
                        url = title_elem.get('href', '')
                        if not url.startswith('http'):
                            url = 'https://www.cnbc.com' + url
                            
                        article = {
                            'title': title,
                            'url': url,
                            'source': 'CNBC',
                            'published_at': datetime.now().isoformat()
                        }
                        
                        if not self.is_duplicate(article):
                            articles.append(article)
                            
        except Exception as e:
            self.logger.error(f"Error scraping CNBC: {str(e)}")
            
        return articles

    def scrape_marketwatch(self) -> List[Dict[str, Any]]:
        """Scrape news from MarketWatch"""
        articles = []
        try:
            response = self.safe_request(self.news_sources['marketwatch'])
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Updated selector for MarketWatch news items
                news_items = soup.find_all('div', {'class': 'article__body'}) or \
                            soup.find_all('div', {'class': 'article__headline'}) or \
                            soup.find_all('h3', {'class': 'article__headline'}) or \
                            soup.find_all('div', {'class': 'article'})
                
                for item in news_items:
                    # Try different possible title and link selectors
                    title_elem = item.find('a', {'class': 'link'}) or item.find('a') or item
                    if title_elem:
                        title = title_elem.text.strip()
                        url = title_elem.get('href', '') if title_elem.name == 'a' else title_elem.find('a').get('href', '')
                        if not url.startswith('http'):
                            url = 'https://www.marketwatch.com' + url
                            
                        article = {
                            'title': title,
                            'url': url,
                            'source': 'MarketWatch',
                            'published_at': datetime.now().isoformat()
                        }
                        
                        if not self.is_duplicate(article):
                            articles.append(article)
                            
        except Exception as e:
            self.logger.error(f"Error scraping MarketWatch: {str(e)}")
            
        return articles

    def scrape_cnn_business(self) -> List[Dict[str, Any]]:
        """Scrape news from CNN Business Markets section"""
        articles = []
        try:
            response = self.safe_request(self.news_sources['cnn_business'])
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try multiple selectors for news items
                news_items = (
                    soup.find_all('article', {'class': 'container__item'}) or  # Article container
                    soup.find_all('div', {'class': 'container__item'}) or      # Alternative container
                    soup.find_all('div', {'class': 'card'})                    # Card container
                )
                
                for item in news_items:
                    try:
                        # Find title and link
                        title_elem = (
                            item.find('span', {'class': 'container__headline-text'}) or  # Headline text
                            item.find('a', {'class': 'container__link'}) or             # Link container
                            item.find('a')                                              # Generic link
                        )
                        
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            url = title_elem.get('href', '')
                            
                            # Handle relative URLs
                            if url.startswith('/'):
                                url = 'https://www.cnn.com' + url
                            elif not url.startswith('http'):
                                url = 'https://www.cnn.com/business' + url
                                
                            article = {
                                'title': title,
                                'url': url,
                                'source': 'CNN Business',
                                'published_at': datetime.now().isoformat()
                            }
                            
                            if not self.is_duplicate(article):
                                articles.append(article)
                                
                    except Exception as e:
                        self.logger.error(f"Error processing CNN Business article: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error scraping CNN Business: {str(e)}")
            
        return articles

    def parse_rss_feed(self, url: str) -> List[Dict[str, Any]]:
        """Parse RSS feed and convert to article format"""
        articles = []
        try:
            # Handle different feed types
            if 'bloomberg.com' in url:
                # Bloomberg uses a sitemap instead of RSS
                response = self.safe_request(url)
                if response and response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'xml')
                    urls = soup.find_all('url')
                    
                    for url_elem in urls:
                        try:
                            loc = url_elem.find('loc')
                            lastmod = url_elem.find('lastmod')
                            if loc:
                                article = {
                                    'title': loc.text.split('/')[-1].replace('-', ' ').title(),
                                    'url': loc.text,
                                    'source': 'Bloomberg',
                                    'published_at': lastmod.text if lastmod else datetime.now().isoformat()
                                }
                                if not self.is_duplicate(article):
                                    articles.append(article)
                        except Exception as e:
                            self.logger.error(f"Error processing Bloomberg sitemap entry: {str(e)}")
                            continue
                            
            elif 'reuters.com' in url:
                # Reuters requires a different approach
                response = self.safe_request(url)
                if response and response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    news_items = soup.find_all('article') or soup.find_all('div', {'class': 'story'})
                    
                    for item in news_items:
                        try:
                            title_elem = item.find('h3') or item.find('a')
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                link = title_elem.get('href', '') if title_elem.name == 'a' else title_elem.find('a').get('href', '')
                                
                                if link and not link.startswith('http'):
                                    link = 'https://www.reuters.com' + link
                                    
                                article = {
                                    'title': title,
                                    'url': link,
                                    'source': 'Reuters',
                                    'published_at': datetime.now().isoformat()
                                }
                                if not self.is_duplicate(article):
                                    articles.append(article)
                        except Exception as e:
                            self.logger.error(f"Error processing Reuters article: {str(e)}")
                            continue
                            
            else:
                # Standard RSS feed parsing
                feed = feedparser.parse(url)
                if feed.bozo and feed.bozo_exception:
                    self.logger.error(f"Error parsing RSS feed {url}: {feed.bozo_exception}")
                    return articles
                    
                for entry in feed.entries:
                    try:
                        # Get source from feed title or URL
                        source = feed.feed.get('title', '')
                        if not source:
                            parsed_url = urlparse(url)
                            source = parsed_url.netloc.split('.')[-2].title()
                        
                        article = {
                            'title': entry.get('title', '').strip(),
                            'description': entry.get('description', entry.get('summary', '')).strip(),
                            'url': entry.get('link', ''),
                            'source': source,
                            'published_at': entry.get('published', entry.get('updated', datetime.now().isoformat()))
                        }
                        
                        # Validate required fields
                        if article['title'] and article['url']:
                            if not self.is_duplicate(article):
                                articles.append(article)
                    except Exception as e:
                        self.logger.error(f"Error processing RSS entry: {str(e)}")
                        continue
                    
        except Exception as e:
            self.logger.error(f"Error parsing feed {url}: {str(e)}")
            
        return articles

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract company names, tickers, and sectors from text"""
        entities = {
            'companies': [],
            'sectors': [],
            'tickers': []
        }
        
        # Extract tickers (e.g., $AAPL, AAPL)
        tickers = re.findall(r'[$]?[A-Z]{1,5}(?:\.[A-Z]{1,2})?', text)
        entities['tickers'] = [tick.replace('$', '') for tick in tickers]
        
        # Extract company names and sectors using NLTK
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        
        # Extract named entities
        named_entities = nltk.chunk.ne_chunk(tagged)
        
        for entity in named_entities:
            if hasattr(entity, 'label'):
                if entity.label() == 'ORGANIZATION':
                    company_name = ' '.join([e[0] for e in entity])
                    if company_name in self.company_keywords:
                        entities['companies'].append(company_name)
                elif entity.label() == 'GPE':
                    sector_name = ' '.join([e[0] for e in entity])
                    if sector_name in self.sector_keywords:
                        entities['sectors'].append(sector_name)
        
        return entities

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text using VADER"""
        scores = self.sia.polarity_scores(text)
        
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def _setup_session(self) -> requests.Session:
        """Setup session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_random_headers(self) -> Dict[str, str]:
        """Get random user agent and headers"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }

    def safe_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Make a safe request with error handling and proxy rotation"""
        try:
            response = self.session.get(
                url,
                headers=self.get_random_headers(),
                timeout=timeout
            )
            response.raise_for_status()
            time.sleep(random.uniform(1, 3))  # Random delay between requests
            return response
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def is_duplicate(self, article: Dict[str, Any]) -> bool:
        """Check if article is a duplicate using title and URL"""
        # Create a unique hash for the article
        article_hash = hashlib.md5(
            (article['title'] + article['url']).encode()
        ).hexdigest()
        
        if article_hash in self.article_cache:
            return True
            
        # Add to cache
        self.article_cache.add(article_hash)
        return False

    def _load_company_keywords(self) -> Set[str]:
        """Load company names and tickers"""
        companies = set()
        try:
            # Get S&P 500 companies
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(sp500_url)
            if tables and len(tables) > 0:
                sp500 = tables[0]
                if 'Symbol' in sp500.columns and 'Security' in sp500.columns:
                    companies.update(sp500['Symbol'].dropna().tolist())
                    companies.update(sp500['Security'].dropna().tolist())
            
            # Add more companies from other indices
            nasdaq_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(nasdaq_url)
            if tables and len(tables) > 0:
                nasdaq = tables[0]
                if 'Ticker' in nasdaq.columns and 'Company' in nasdaq.columns:
                    companies.update(nasdaq['Ticker'].dropna().tolist())
                    companies.update(nasdaq['Company'].dropna().tolist())
            
            # Add some common stock symbols manually as fallback
            common_symbols = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'}
            companies.update(common_symbols)
            
            self.logger.info(f"Loaded {len(companies)} company keywords")
            return companies
            
        except Exception as e:
            self.logger.error(f"Error loading company keywords: {str(e)}")
            # Return a basic set of common stock symbols as fallback
            return {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'}

    def _load_sector_keywords(self) -> Set[str]:
        """Load sector and industry keywords"""
        sectors = {
            'Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial',
            'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication',
            'Semiconductor', 'Software', 'Biotech', 'Pharmaceutical', 'Banking',
            'Insurance', 'Retail', 'Automotive', 'Aerospace', 'Defense',
            'Oil', 'Gas', 'Renewable', 'Mining', 'Construction'
        }
        return sectors