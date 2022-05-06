import re
 
# 将匹配的数字乘以 2
def map(matched):
    gram2ab, ab2gram = load_abbre()
    return ab2gram[matched.group(0)]
 
def unpackAbbre(sentence):
	return re.sub('abbre\d+', map, sentence)

def load_abbre():
    bi_set = [
        'regulation of',
        'response to',
        'involved in',
        'signaling pathway',
        'biosynthetic process',
        'metabolic process',
        'catabolic process',
        'cell differentiation',
        'receptor activity',
        'receptor binding',
        'transporter activity',
        'receptor signaling',
        'cell proliferation',
        't cell',
        'channel activity',
        'kinase activity',
        'protein localization',
        'oxidoreductase activity',
        'muscle cell',
        'polymerase ii',
        'cell migration',
        'plasma membrane'
	]
    tri_set = [
        'positive regulation of',
        'negative regulation of',
        'cellular response to',
        'transmembrane transporter activity',
        'rna polymerase ii'
    ]
    abbre_set = tri_set + bi_set
    ab2gram = {}
    gram2ab = {}
    for idx, item in enumerate(abbre_set):
        gram2ab[item] = "abbre" + str(idx)  # tuple 2 abbre
        ab2gram["abbre" + str(idx)] = item          # abbre 2 list
    return gram2ab, ab2gram

if __name__ == '__main__':
    s = 'i hh abbre1 abbre11 you abbre5'
    print(unpackAbbre(s))



