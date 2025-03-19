with open('spotify_article.txt', 'r', encoding='utf-8') as f:
    article_text = f.read()

print("Wikipedia Article Text:")
print(article_text)

print("\nSee Also Links:")
with open('see_also_links.txt', 'r', encoding='utf-8') as f:
    see_also_links = f.read()

print(see_also_links)
