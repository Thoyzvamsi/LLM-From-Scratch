from bs4 import BeautifulSoup as bs
from data_pipeline.data_handling import Data_handling
import requests
import re
from pathlib import Path
import json
import time

data_path = "data\data.txt"
urls_path = r"data\urls_file.json"
api_url = "https://en.wikipedia.org/w/api.php"
params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 10,
        "format": "json"
    }

# Enter the number of articles multiplies with 10*10
num = 500
headers = {
        "User-Agent": "Kairo_Scrapper/1.0 (learning project)"
    }

 # --- Basic web scrapper for Text data in Wikipedia articles ---

class Crawler:
    def __init__(self ,file_urls):
        self.crawled = set(file_urls)
        self.headers = headers 
        

    def crawler(self,current_url):
        try:
            html = requests.get(current_url, headers=headers).text
        except Exception:
            print(f"Error in {Exception}")

        self.crawled.add(current_url)  
        # Scrapes the data
        self.scrapper(current_url)

        return list(self.crawled)
    
        
    def scrapper(self,url):
        # --- It appends the text data to data.txt file ---
        html = requests.get(url,headers=headers).text

        soup = bs(html,"html.parser")
        content = soup.find("div", class_="mw-parser-output")
        
        if content:
            text = content.get_text(" ", strip=True)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text) 

            #Replace them with space to prevent errors
            text = re.sub(
                r"[\u200e\u200f\u2028\u2029]",
                " ",
                text
            )
            print(url)
            print("Text length:", len(text))
            print("Word count:", len(text.split()))
            print("Quality:", Data_handling().data_quality_filter(text))
            
            if Data_handling().data_quality_filter(text):
                with open(data_path, "a", encoding="utf-8") as f:
                    f.write(text)  
                    f.write("\n")


def main():
    urls = []
    if not Path(urls_path).exists():
        with open(urls_path, "w") as f:
            json.dump([], f)
    
    with open(urls_path,"r") as f:
        file_urls = json.load(f)
    
    Crawl = Crawler(file_urls)

    for i in range(num):
        response = requests.get(
            api_url,
            params=params,
            headers=headers
        )
        print(response.status_code)
        data = response.json()

        for article in data["query"]["random"]:
            title = article["title"]
            url = "https://en.wikipedia.org/wiki/"+ title.replace(" ", "_")
            print("Found article:", url)
            urls.append(url)

        if i % 10 == 0:
            time.sleep(5)

    for url_in_list in urls:
        # checks if Already crawled 
        if url_in_list in file_urls:
            continue
        crawled_urls = Crawl.crawler(url_in_list)

    with open(urls_path, "w") as f:
        json.dump(list(crawled_urls), f, indent=4)
    

if __name__ == "__main__":
    main()