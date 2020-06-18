import os
import random
import re
import sys
import copy
import pandas

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    l = len(corpus[page])
    if l == 0:
        l = len(corpus)
    damp = 1.0 * damping_factor / (l * 1.0)
    oneDamp = (1.0 - damping_factor) / (len(corpus) * 1.0)
    probDistro = dict()
    for p in corpus:
        probDistro[p] = 0.0
        if corpus[page] != None:
            if p in corpus[page]:
                probDistro[p] += damp
            probDistro[p] += oneDamp
        else:
            probDistro[p] += damp + oneDamp

    return probDistro

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRank = dict()
    samples = dict()
    #choose a random page in corpus
    pg = random.choice(list(corpus.keys()))
    while len(corpus[pg]) == 0:
        pg = random.choice(list(corpus.keys()))
    page = random.choice(list(corpus[pg]))
    samples[page] = 1
    for i in corpus:
        samples[i] = 0
    #iterate and generate n samples
    for i in range(n):
        probability = transition_model(corpus, page, damping_factor)
        rp = random.random()
        #get random number range associated with each page in transition model
        count = 0.0
        for key in probability:
            if rp <= probability[key] + count:
                samples[key] += 1
                break
            count += probability[key]
    for key in samples:
        pageRank[key] = samples[key] / (n * 1.0)
    s = 0.0 
    for p in pageRank:
        s += pageRank[p]
    print(f"Sum: {s}")
    return pageRank



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    print(corpus)
    pageRank = dict()
    oldPageRank = dict()
    newPageRank = dict()
    n = len(corpus.keys())
    for c in corpus:
        pageRank[c] = float(1 / n)
        oldPageRank[c] = float(10)
        newPageRank[c] = 0.0
        if len(corpus[c]) == 0:
            for i in corpus:
                corpus[c].add(i)
    print(corpus)

    #C = {x: A[x] - B[x] for x in A}
    while all(not r < .001 for r in {x: abs(oldPageRank[x] - pageRank[x]) for x in pageRank}.values()):
        oldPageRank = copy.deepcopy(pageRank)
        for p in corpus:
            parents = pageParents(corpus, p)
            summ = float(0)
            for i in parents:
                l = len(corpus[i])
                summ += pageRank[i] / l
            newPageRank[p] = ((1.0-damping_factor) / n) + (damping_factor * summ)
        for p in newPageRank:
            pageRank[p] = newPageRank[p]
    s = sum(pageRank.values())
    print(f"Sum: {s}")
    return pageRank
    
def pageParents(corpus, page):
    parents = {key for key, values in corpus.items() if page in values or len(values) == 0}
    return parents

if __name__ == "__main__":
    main()
