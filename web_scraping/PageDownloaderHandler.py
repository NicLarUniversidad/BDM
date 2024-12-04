import requests
from bs4 import BeautifulSoup

class PageDownloadHandler(object):
    def download(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0'}
        page = requests.get(url, headers=headers)
        content = page.content
        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("div", {"class": "table-container"})
        history = dict()
        for table in tables:
            for tag in table.children:
                td = tag.find_all("td")
                row = []
                count = 0
                for t in td:
                    count += 1
                    if "Dividendo" in t.text or "Divisiones" in t.text:
                        row = []
                        count = 0
                    else:
                        if count % 7 == 0:
                            history[row[0]] = (row[1], row[2], row[3], row[4], row[5])
                            row = []
                        else:
                            row.append(t.text)
        return history