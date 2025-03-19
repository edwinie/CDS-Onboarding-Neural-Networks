import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://en.wikipedia.org/wiki/Spotify'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'lxml')
content = soup.find('div', id='mw-content-text')
readable_text = ""
for tag in content.find_all(['p', 'h2', 'h3']):
    text = tag.get_text(strip=True)
    if text:
        readable_text += text + "\n\n"

with open('spotify_article.txt', 'w', encoding='utf-8') as f:
    f.write(readable_text)

see_also_links = []
see_also_header = soup.find('h2', id='See_also')
if see_also_header:
    parent_div = see_also_header.parent
    ul = parent_div.find_next_sibling('ul')
    if ul:
        for li in ul.find_all('li'):
            a = li.find('a', href=True)
            if a:
                see_also_links.append({
                    'title': a.text.strip(),
                    'url': 'https://en.wikipedia.org' + a['href']
                })

with open('see_also_links.txt', 'w', encoding='utf-8') as f:
    for link in see_also_links:
        f.write(f"{link['title']}: {link['url']}\n")