from src.news_analyst.news_analyzer import NewsAnalyzer
import time
import json
from datetime import datetime

def test_news_scraping():
    print("Initializing NewsAnalyzer...")
    analyzer = NewsAnalyzer()
    
    print("\nTesting web scraping sources...")
    
    # Test Yahoo Finance
    print("\nTesting Yahoo Finance scraping...")
    articles = analyzer.scrape_yahoo_finance()
    print(f"Found {len(articles)} articles from Yahoo Finance")
    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
    
    # Test CNBC
    print("\nTesting CNBC scraping...")
    articles = analyzer.scrape_cnbc()
    print(f"Found {len(articles)} articles from CNBC")
    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
    
    # Test MarketWatch
    print("\nTesting MarketWatch scraping...")
    articles = analyzer.scrape_marketwatch()
    print(f"Found {len(articles)} articles from MarketWatch")
    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
    
    # Test CNN Business
    print("\nTesting CNN Business scraping...")
    articles = analyzer.scrape_cnn_business()
    print(f"Found {len(articles)} articles from CNN Business")
    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
    
    print("\nTesting RSS feeds...")
    
    # Test Seeking Alpha RSS feed
    print("\nTesting Seeking Alpha RSS feed...")
    articles = analyzer.parse_rss_feed(analyzer.news_sources['seeking_alpha'])
    print(f"Found {len(articles)} articles from Seeking Alpha")
    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
    
    # Test Financial Times RSS feed
    print("\nTesting Financial Times RSS feed...")
    articles = analyzer.parse_rss_feed(analyzer.news_sources['ft'])
    print(f"Found {len(articles)} articles from Financial Times")
    if articles:
        print("\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
    
    # Test entity extraction
    print("\nTesting entity extraction...")
    sample_article = {
        'title': "Stocks making the biggest moves midday: U.S. Steel, Tesla, Dollar Tree, Apple and more",
        'url': "https://www.cnbc.com/2025/04/07/stocks-making-the-biggest-moves-midday-x-tsla-dltr-aapl-and-more.html"
    }
    print("\nSample article entities:")
    print(f"Title: {sample_article['title']}")
    entities = analyzer.extract_entities(sample_article['title'])
    print("Extracted entities:")
    print(json.dumps(entities, indent=2))
    
    # Test sentiment analysis
    print("\nTesting sentiment analysis...")
    print("\nSentiment analysis for top 3 articles:")
    sample_articles = [
        {
            'source': 'CNBC',
            'title': "Stocks making the biggest moves midday: U.S. Steel, Tesla, Dollar Tree, Apple and more"
        },
        {
            'source': 'CNBC',
            'title': "CEOs think the U.S. is 'probably in a recession right now,' says BlackRock's Larry Fink"
        },
        {
            'source': 'CNBC',
            'title': "Why the 'Magnificent Seven' could be hit harder than rest of S&P 500 on Trump's tariffs"
        }
    ]
    
    for article in sample_articles:
        print(f"\nArticle from {article['source']}:")
        print(f"Title: {article['title']}")
        print(f"Sentiment: {analyzer.analyze_sentiment(article['title'])}")
    
    # Save sample articles to file
    with open('sample_articles.json', 'w') as f:
        json.dump(sample_articles, f, indent=2)
    print("\nSaved sample articles to sample_articles.json")

if __name__ == "__main__":
    test_news_scraping()