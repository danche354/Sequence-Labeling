'''
for loading letter-3-gram
'''

l3g_dict = {}

with open('../preprocessing/l3g.txt') as f:
    l3g_list = f.read().strip().split('\n')
    for i, each in enumerate(l3g_list):
        l3g_dict[each] = i
