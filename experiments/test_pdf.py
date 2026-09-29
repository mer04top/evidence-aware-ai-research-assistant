import pymupdf

doc = pymupdf.open('data/papers/lewis.pdf')

for page in doc:
    print(page.get_text()[:300])
    print('---')

# messy
# idfk whats going on
# not fixing this now
# nvm i think it's fine, just pictures