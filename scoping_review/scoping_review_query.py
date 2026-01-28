import re
import requests
import urllib.parse
from bs4 import BeautifulSoup
from collections import defaultdict


# Machine learning and deep learning methods for post-transplant infection risk prediction: a scoping review.
G_METHODS = {
    0: [
        "deep learning",
        "machine learning",
        "language model",  # adds no paper in pubmed search
    ],
    1: [
        "artificial intelligence",
        "transformer",  # adds no paper in pubmed search
    ],   
    2: [
        "neural network*",
        "bayesian network*",
    ],
    3: [
        "XGBoost",  # adds no paper in pubmed search
        "decision tree*",
        "random forest*",
    ],
    4: [
        "SVM",  # adds no paper in pubmed search
        "CNN",  # adds no paper in pubmed search
        "GNN", # adds no paper in pubmed search
        "LSTM",  # adds no paper in pubmed search
    ],
}

G_STUDY_DOMAINS = {
    0: [
        "transplant*",
        "graft*",
    ],
}

G_STUDY_TOPICS = {
    0: [
        "infection*",
        "infectious disease",  # adds no paper in pubmed search
        "sepsis",
    ],
    1: [
        "virus",
        "bacteria",
        "fungus",  # adds no paper in pubmed search
        # "parasite",
    ],
}

G_OUTCOMES = {
    0: [
        "risk",
        "prediction",
        "prognosis",
        "detection",
    ],
    1: [
        "assessment",
        "classification",
        "categorization",  # adds no paper in pubmed search
    ],
}

# QUERY_GROUP_ORDERS = [
#     {"method": 0, "topic": 0, "outcome": 0},
#     {"method": 0, "topic": 0, "outcome": 1},
#     {"method": 0, "topic": 1, "outcome": 0},
#     {"method": 0, "topic": 1, "outcome": 1},
#     {"method": 1, "topic": 0, "outcome": 0},
#     {"method": 1, "topic": 0, "outcome": 1},
#     {"method": 1, "topic": 1, "outcome": 0},
#     {"method": 1, "topic": 1, "outcome": 1},
# ]

QUERY_GROUP_ORDERS = [
    {"method": 0},
    {"method": 1},
    {"method": 2},
    {"method": 3},
    {"method": 4},
]


def main():
    """ Main function to generate a pubmed query and convert it to a scholar query
    """
    # Pubmed query generation
    pubmed_query = create_pubmed_query({})
    pubmed_query_with_date_and_language = f'{pubmed_query} AND (english[lang]) AND ("2015/01/01"[Date - Publication] : "2025/04/30"[Date - Publication])'
    num_papers = get_num_pubmed_papers(pubmed_query_with_date_and_language)
    print(f"\nPubmed query: {pubmed_query_with_date_and_language}")
    print(f"\nNumber of identified PubMed papers: {num_papers}")
    
    # Scholar query generation (step by step)
    scholar_query = pubmed_to_scholar(pubmed_query, shorten=True)
    print(f"\nFull scholar query ({len(scholar_query)} characters): {scholar_query}")
    for query_group_order in QUERY_GROUP_ORDERS:
        split_pubmed_query = create_pubmed_query(query_group_order)
        split_scholar_query = pubmed_to_scholar(split_pubmed_query, shorten=True)
        print(f"\nSplit scholar query ({len(split_scholar_query)} characters): {split_scholar_query}")
        

def create_pubmed_query(
    query_group_order: dict[str, int],
):
    """ Generate a pubmed search query based on predefined terms
    """
    query_parts = []
    for group_name, group_terms in zip(
        ["method", "domain", "topic", "outcome"],
        [G_METHODS, G_STUDY_DOMAINS, G_STUDY_TOPICS, G_OUTCOMES],
    ):
        if group_name in query_group_order:
            group = group_terms[query_group_order[group_name]]
        else:
            group = [term for group in group_terms.values() for term in group]
            if query_group_order == {}:
                print("\nTerms for group " + group_name + ":")
                print('"' + '" OR "'.join(group) + '"')
                
        formatted_terms = [f'"{term}"[Title/Abstract]' for term in group]
        query_parts.append("(" + " OR ".join(formatted_terms) + ")")
        
    return " AND ".join(query_parts).replace("\n", " ")


def get_num_pubmed_papers(query):
    """ Fetch the number of identified papers from pubmed based on the query
    """
    # Extract response from pubmed query
    encoded_query = urllib.parse.quote(query)
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded_query}"
    response = requests.get(pubmed_url)
    response.raise_for_status()
    
    # Extract number of identified papers from the response
    soup = BeautifulSoup(response.text, "html.parser")
    script_tags = soup.find_all("script")
    for script in script_tags:
        if script.string and "totalResults" in script.string:
            match = re.search(r'totalResults:\s*parseInt\("(\d+)"', script.string)
            if match:
                return int(match.group(1))
                
    # In case nothing was found
    print("Could not retrieve the total number of papers.")
    return None


def pubmed_to_scholar(pubmed_query: str, shorten: bool=False):
    """ ...
    """
    query = re.sub(r'\[Title/Abstract\]', '', pubmed_query)
    query = query.replace("*", "").strip()  # apparently scholar does not support "*"
    query = re.sub(r'\s+', ' ', query)
    
    if shorten:
        query = shorten_scholar_query(query)
    
    return query


def merge_common_terms(terms):
    """ Group terms by identifying and merging common words.
        For now, works poorly for terms with more than two words.
    """
    # Build word pairs
    terms_split_by_words = []
    first_word_groups = defaultdict(set)
    second_word_groups = defaultdict(set)
    for term in terms:
        words = term.split(" ")
        first, second = words[0], " ".join(words[1:])
        first_word_groups[first].add(second)
        if second != "":
            second_word_groups[second].add(first)
        terms_split_by_words.append([first, second])
    
    # Identify terms to be merged
    merged_terms = dict()  # to have a set that keeps the order
    for first, second in terms_split_by_words:
        if len(first_word_groups[first]) > 1:
            merged = "|".join(sorted(first_word_groups[first]))
            merged_terms.update({f"{first} {merged}": None})
        elif len(second_word_groups[second]) > 1:
            merged = "|".join(sorted(second_word_groups[second]))
            merged_terms.update({f"{merged} {second}": None})
        else:
            merged_terms.update({f"{first} {second}".strip(): None})  # strip because second may be ""
    
    return [f'"{term}"' if len(term.split()) > 1 else term for term in merged_terms.keys()]


def shorten_scholar_query(query: str) -> str:
    """ Shorten a scholar query by merging terms
    """
    new_query_terms = []
    for or_group in re.findall(r'\((.*?)\)', query):
        terms = [t.strip('"') for t in or_group.split(" OR ")]
        # terms = [t for t in or_group.split(" OR ")]
        merged_terms = merge_common_terms(terms)
        new_query_terms.append(f'({" OR ".join(merged_terms)})')
            
    return " AND ".join(new_query_terms)


if __name__ == "__main__":
    main()