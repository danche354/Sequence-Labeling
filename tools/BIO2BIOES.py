dataset = ['eng.train', 'eng.testa', 'eng.testb']
ner_BIOES = '../dataset/ner_BIOES/'


for data in dataset:
    result = open(ner_BIOES+data, 'w')
    with open('../dataset/ner/'+data, 'r') as f:
        sentences = f.read().strip().split('\n\n')
        for i, sentence in enumerate(sentences):
            tokens = sentence.split('\n')
            length = len(tokens)
            for i, token in enumerate(tokens):
                token_split = token.split()
                ner_label = token_split[-1]
                if i==0:
                    if length==1:
                        if ner_label.startswith("I-"):
                            token_split[-1] = "S-"+ner_label[2:]
                    else:
                        suf = tokens[i+1].split()[-1]
                        if ner_label.startswith("I-"):
                            if ner_label==suf:
                                token_split[-1] = "B-"+ner_label[2:]
                            else:
                                token_split[-1] = "S-"+ner_label[2:]
                elif i==length-1:
                    pre = tokens[i-1].split()[-1]
                    if ner_label.startswith('I-'):
                        if ner_label[2:]==pre[2:]:
                            token_split[-1] = "E-"+ner_label[2:]
                        else:
                            token_split[-1] = "S-"+ner_label[2:]
                    elif ner_label.startswith('B-'):
                        token_split[-1] = "S-"+ner_label[2:]
                else:
                    pre = tokens[i-1].split()[-1]
                    suf = tokens[i+1].split()[-1]
                    if ner_label.startswith("I-"):
                        if ner_label[2:]==pre[2:] and ner_label!=suf:
                            token_split[-1] = "E-"+ner_label[2:]
                        elif ner_label[2:]!=pre[2:]:
                            if ner_label!=suf:
                                token_split[-1] = "S-"+ner_label[2:]
                            else:
                                token_split[-1] = "B-"+ner_label[2:]
                    elif ner_label.startswith("B-"):
                        if ner_label[2:]!=suf[2:] or ner_label==suf:
                            token_split[-1] = "S-"+ner_label[2:]
                result.write(" ".join(token_split)+"\n")
            result.write("\n")




                




