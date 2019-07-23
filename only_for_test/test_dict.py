

file = open('../data/hdu/Knowledge_Problem/hdu_knowledgeName2knowledgeId.txt', 'r',
            encoding='utf-8')
file = eval(file.read())
knowledgeLabel = list(file.keys())
print(knowledgeLabel)