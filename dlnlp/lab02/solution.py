import nltk
import spacy
from nltk import CFG

grammar = CFG.fromstring("""
S -> NP VP
NP -> NA N
VP -> VP Adv
VP -> Aux V
NP -> NP PP
NP -> Det N
PP -> P NP
VP -> Aux V
NP -> NP Conj NP
NP -> Adj N
VP -> V NP
VP -> V NP
VP -> VP CP
CP -> CP NP
Conj -> P Conj   
CP -> P Conj
NA -> 'flying'
V -> 'be' | 'flying' | 'loves'
Adv -> 'dangerous'
Adj -> 'dangerous'
Det -> 'the'
Aux -> 'were' | 'can'
P -> 'of' | 'more'
Conj -> 'and' | 'then' | 'than'
N -> 'parents' | 'bride' | 'groom' | 'planes'
""")

# Create a parser using the grammar
parser = nltk.ChartParser(grammar)

# Sentence to parse
sent1 = "Flying planes can be dangerous".lower().split()
sent2 = "The parents of the bride and the groom were flying".lower().split()
sent3 = "The groom loves dangerous planes more than the bride".lower().split()

# Parse the sentence using the CKY algorithm
for sentence in [sent1, sent2, sent3]:
    print("#" * 12, sentence, "#" * 12)
    for tree in parser.parse(sentence):
        print(tree)
        tree.pretty_print()

### point 2


nlp_model = spacy.load("en_core_web_sm")

for sentence in [sent1, sent2, sent3]:
    print("#" * 12, sentence, "#" * 12)
    doc = nlp_model(" ".join(sentence))
    for token in doc:
        print(f"{token.text} ({token.dep_}) <- {token.head.text}")


## POINT 3


class Application:
    """
    Legal document analyzer

    A tool that analyzes legal contracts and documents to detect ambiguities, redundant clauses or problematic phrasing using syntactic and dependency parsing.

    The app parses complex legal sentences to identify relationships between clauses, detect ambiguous phrasing, and provide suggestions for simplifying legal language.

    It highlights long-distance dependencies between clauses that could cause confusion, making contracts easier to understand.

    Why?
    Legal documents often have complex sentence structures with many dependencies, and parsing these dependencies is crucial for detecting ambiguities or errors that could lead to legal disputes.

    Rigorous Structure: Legal documents tend to follow structured grammar and formal language, which makes them ideal for syntactic parsing. CKY parsing can break down complex legal sentences and identify relationships between clauses.

    Ambiguity Detection: In legal documents, ambiguity can lead to misinterpretation. CKY parsing can generate multiple parse trees for sentences with ambiguous structures, helping to detect parts of the document where multiple interpretations are possible.

    Clause and Contract Structure Analysis: CKY parsers can extract nested clauses or identify the relationships between different sections of a legal document, making it easier to spot dependencies or conflicts between clauses.

    """


class CNN:
    """
    # Chomsky Normal Form:
    S -> NP VP
    NP -> NA N | Det N | Adj N | NP X1 | NP PP
    X1 -> Conj NP
    VP -> Aux | V NP | X2 Adv | X3 CP
    X2 -> VP
    X3 -> VP
    CP -> X4 NP
    X4 -> CP
    PP -> P NP
    Conj -> P Conj | 'and' | 'then' | 'than'
    X5 -> NA
    V -> 'be' | 'flying' | 'loves'
    Adv -> 'dangerous'
    Adj -> 'dangerous'
    Det -> 'the'
    Aux -> 'were' | 'can'
    P -> 'of' | 'more'
    N -> 'parents' | 'bride' | 'groom' | 'planes'
    """
