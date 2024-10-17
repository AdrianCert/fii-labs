## Phrase 1

(S "Flying planes can be dangerous")
    (NP "Flying planes")
        (NA "Flying")
        (N "planes")
    (VP "can be dangerous")
        (VP "can be")
            (Aux "can")
            (V "be")    
        (Adv "dangerous")

S -> NP VP
NP -> NA N
VP -> VP Adv
VP -> Aux V
N -> 'planes'
NA -> 'flying'
Aux -> 'can'
V -> 'be'
Adv -> 'dangerous'

## Phrase 2

(S "The parents of the bride and the groom were flying")
    (NP "The parents of the bride and the groom")
        (NP "The parents of the bride")
            (NP "The parents")
                (Det "the")
                (N "parents")
            (PP "of the bride")
                (P "of")
                (NP "the bride")
                    (Det "the")
                    (N "bride")
        (Conj "and")
        (NP "the groom")
            (Det "the")
            (N "groom")
    (VP "were flying")
        (Aux "were")
        (V "flying")
        
(S "The parents of the bride and the groom were flying")
    (NP "The parents of the bride and the groom")
        (NP "The parents")
            (Det "the")
            (N "parents")
        (PP "of the bride and the groom")
            (P "of")
            (NP "the bride and the groom")
                (NP "the bride")
                    (Det "the")
                    (N "bride")
                (Conj "and")
                (NP "the groom")
                    (Det "the")
                    (N "groom")
    (VP "were flying")
        (Aux "were")
        (V "flying")

S -> NP VP
NP -> NP PP
NP -> Det N
PP -> P NP
VP -> Aux V
NP -> NP Conj NP
V -> 'flying'
Det -> 'the'
Aux -> 'were'
P -> 'of'
Conj -> 'and'
N -> 'parents' | 'bride' | 'groom'


## Phrase 3

(S "The groom loves dangerous planes more than the bride")
    (NP "The groom")                     
        (Det "the")
        (N "groom")
    (VP "loves dangerous planes more than the bride")
        (VP "loves dangerous planes")
            (V "loves")
            (NP "dangerous planes")
                (Adj "dangerous")
                (N "planes")
        (CP "more than the bride")
            (CP "more than")
                (P "more")
                (Conj "than")
            (NP "the bride")
                (Det "the")
                (N "bride")


(S "The groom loves dangerous planes more than the bride")
    (NP "The groom")                     
        (Det "the")
        (N "groom")
    (VP "loves dangerous planes more than the bride")
        (V "loves")
        (NP "dangerous planes more than the bride")
            (NP "dangerous planes")
                (Adj "dangerous")
                (N "planes")
            (Conj "more than")
                (P "more")
                (Conj "than")
            (NP "the bride")
                (Det "the")
                (N "bride") 

S -> NP VP
VP -> V NP
VP -> VP CP
CP -> CP NP
CP -> P Conj
NP -> Det N
NP -> Adj N
Adj -> 'dangerous'
N -> 'planes'
NP -> NP Conj NP
P -> 'more'
Conj -> 'then'
Conj -> P Conj
