from bs4 import BeautifulSoup as bs
from data_pipeline.data_handling import Data_handling
import requests
import re
from pathlib import Path
import json
import time

# --- Basic web scrapper for Text data in Wikipedia articles ---

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

# Enter the number of articles multiplies with 10
number_of_articles = 50
headers = {
        "User-Agent": "Kairo_Scrapper/1.0 (learning project)"
    }


class Crawler:
    def crawler(self,current_url):
        """ 
            Opening and rewritting it every time because when ever requests 
            are out of limit the urls already crawled will get stored 

        """
        with open(urls_path,"r") as f:
            file_urls = json.load(f)
        
        file_urls = set(file_urls)
        
        if current_url in file_urls:
            print("Already crawled")
        else:
            file_urls.add(current_url)  
            # Scrapes the data
            self.scrapper(current_url)

            with open(urls_path, "w") as f:
                json.dump(list(file_urls), f, indent=4)


    def scrapper(self,url):
        # --- It appends the text data to data.txt file ---
        try:
            html = requests.get(url, headers=headers).text
        except Exception:
            print(f"Error {Exception}")

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
    if not Path(urls_path).exists():  # Create the file if doesn't Exits
        with open(urls_path, "w") as f:
            json.dump([], f)
    
    Crawl = Crawler()

    for i in range(number_of_articles):
        response = requests.get(
            api_url,
            params=params,
            headers=headers
        )
        data = response.json()

        for article in data["query"]["random"]:
            title = article["title"]
            url = "https://en.wikipedia.org/wiki/"+ title.replace(" ", "_")
            print("Found article:", url)
            urls.append(url)

        if i % 10 == 0: # gives small breathing window for requests
            time.sleep(2)

    for url_in_list in urls:
        Crawl.crawler(url_in_list)


if __name__ == "__main__":
    main()