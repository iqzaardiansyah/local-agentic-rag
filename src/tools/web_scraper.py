import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def read_webpage(url: str) -> str:
    """
    Scrape and read the text content of a webpage.
    Use this after doing a web_search if you need to read the full article or documentation from a specific URL.
    """
    try:
        # Use a standard user-agent to avoid basic blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator='\n')
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit to first 10,000 characters to avoid blowing up the LLM context limit
        if len(text) > 10000:
            text = text[:10000] + "\n...[Content truncated due to length]..."
            
        return text
    except Exception as e:
        return f"Error reading webpage {url}: {str(e)}"
