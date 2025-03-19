import os

def test_files_exist():
    assert os.path.exists('spotify_article.txt'), "Article text file missing!"
    assert os.path.exists('see_also_links.txt'), "See-also links file missing!"

def test_files_content():
    with open('spotify_article.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        assert len(content.strip()) > 0, "Article text file is empty!"

    with open('see_also_links.txt', 'r', encoding='utf-8') as f:
        links = f.readlines()
        assert len(links) > 0, "See-also links file is empty!"

if __name__ == "__main__":
    test_files_exist()
    test_files_content()
    print("All tests passed successfully!")
