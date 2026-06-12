from bs4 import BeautifulSoup as bs
import requests
import re
from urllib.parse import urljoin

class Scrapping:
    def __init__(self ,path):
        self.path = path

    def scrapper(self,current_url):
        html = requests.get(current_url).text

        soup = bs(html,"html.parser")
        temp_urls = []

        self.data_collector(current_url)

        for a in soup.find_all("a",href=True):
            href = a["href"]
            if (
                href.startswith("#")
                or href.startswith("javascript:")
                or href.startswith("mailto:")
            ):
                continue

            url = urljoin(current_url, href)
            temp_urls.append(url)
        
        if temp_urls is None:
            return
        
        for temp_url in temp_urls:
            self.data_collector(temp_url)
        

    def data_collector(self,url):
        html = requests.get(url).text

        soup = bs(html,"html.parser")

        text = soup.get_text(separator=" ",strip=True)
        text = re.sub(r"s\+"," ",text)
        text = re.sub(r"[ \t]+", " ", text)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    
def main():
    urls = ["https://www.rottentomatoes.com/celebrity/leonardo_di_caprio","https://en.wikipedia.org/wiki/House_of_the_Dragon"]
    path = "data.txt"

    scrape = Scrapping(path)

    for url in urls:
        scrape.scrapper(url)

if __name__ == "__main__":
    main()