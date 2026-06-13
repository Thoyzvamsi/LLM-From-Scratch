from bs4 import BeautifulSoup as bs
import requests
import re
from urllib.parse import urljoin

"""
 - Basic web scrapper , Enter you'r urls in list in the main function and run the file

 # Updates:
 - Use function to recursion rather than the loop
 - Collect Good url set 

"""

class Crawler:
    def __init__(self ,path):
        self.path = path
        self.crawled = set()
        self.headers = {
        "User-Agent": "VamsiBot/1.0 (learning project)"
        }

    def crawler(self,current_url):
        """
          - It only scarpes likes from a particular url
        """
        try:
            html = requests.get(current_url, headers=self.headers).text # requesting html
        except Exception:
            return "Error in response"

        soup = bs(html,"html.parser")

        if current_url in self.crawled: # checks if Already crawled 
            return 

        if self.blocked_url_patters(current_url):
            self.crawled.add(current_url)            
        
        self.scrapper(current_url)

        

    def scrapper(self,url):

        # --- It appends the text data to data.txt file ---

        html = requests.get(url,headers=self.headers).text

        soup = bs(html,"html.parser")
        content = soup.find("div", class_="mw-parser-output")
        
        if content:
            text = content.get_text(" ", strip=True)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(
                r"[\u200e\u200f\u2028\u2029]",
                " ",
                text
            ) #Replace them with space to prevent errors

            print(url)
            print(text[:500])
            print("-" * 50)
            print("Text length:", len(text))
            print("Word count:", len(text.split()))
            print("Quality:", self.data_quality_filter(text))
            
            if len(text) > 1500:
                if self.data_quality_filter(text):
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(text)  
                        f.write("\n")

    def blocked_url_patters(self,link):
        # Blocks this kind of patterns 

        blocked_patters = [
                '/sponsors', '/security', '/solutions',
                '/resources', '/events', '/customer-stories',
                '/shop', '/about', '/contact', '/pricing'
            ]
        
        for pattern in blocked_patters:
            if link.endswith(pattern):
                return False
            elif (
                pattern.startswith("#")
                or pattern.startswith("javascript:")
                or pattern.startswith("mailto:")
            ):
               return False

        return True
    
    # Filter the spam or any inconsistencies
    def data_quality_filter(self,text):
        if len(text) < 400 or len(text.split()) < 100:
            return False
        
        alpha_ratio = sum(c.isalpha() for c in text)/len(text)
        if alpha_ratio < 0.6:   # "Hello12389" ratio = 5(chars) / 10(whole word) < 0.6
            return False
        
        sentences = text.split('.')
        uniquness = len(set(sentences))/len(sentences)
        if uniquness < 0.6:     # Same as the above but for sentences
            return False
        
        return True
        
    
def main():
    urls = []
    path = "data.txt"

    Crawl = Crawler(path)
    
    import requests

    api_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 10,
        "format": "json"
    }
    headers = {
        "User-Agent": "VamsiBot/1.0 (learning project)"
    }
    for i in range(10):
        response = requests.get(
            api_url,
            params=params,
            headers=headers
        )

        print(response.status_code)
        data = response.json()
        for article in data["query"]["random"]:
            title = article["title"]

            url = (
                "https://en.wikipedia.org/wiki/"
                + title.replace(" ", "_")
            )
            print("Found article:", url)

            urls.append(url)


    for url_in_list in urls:
        Crawl.crawler(url_in_list)

if __name__ == "__main__":
    main()