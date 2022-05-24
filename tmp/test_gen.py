import pandas as pd
import os



ID2LANDMARK = [
        'Nasion',
        'Sella',
        'Porion',
        'Orbitale',
        'Pterygoid',
        'Basion',
        'ANS',
        'PNS',
        'A point',
        'B point',
        'Protuberance menti',
        'Pogonion',
        'Menton',
        'Go-1 Corpuus left',
        'Go-2 Ramus down',
        'Articulare',
        'R3',
        'R1',
        'Maxilla 1 crown',
        'Maxilla 1 root',
        'Mandible 1 crown',
        'Mandible 1 root',
        'Maxilla 6 distal',
        'Maxilla 6 root',
        'Mandible 6 distal',
        'Mandible 6 root',
        'Glabella',
        'Soft tissue naison',
        'Dorsum of nose',
        'pronasale',
        'Colunella',
        'Subnasalle',
        'Soft tissue A point',
        'Labrale superius',
        'Upper lip',
        'Stms',
        'Upper embrasure',
        'Lower embrasure',
        'Stmi',
        'Lower lip',
        'Labial Inferius',
        'Soft tissue b point',
        'Soft tissue pogonion',
        'Soft tissue Gnathion',
        'Soft tissue Menton',
        'Cervical point',  
    ]

import numpy as np

data = np.load('./data_20220105.npy')
df = pd.read_csv('./label_20220105.csv',index_col=0)


size = 4
anno = './Annotations/'
if not os.path.exists(anno):
    os.mkdir(anno)

ret_str = """<annotation>
    <folder>Ceph20220105</folder>
"""
for idx, c in df.iterrows():
    
    id = c['ID']
    fstr = f'{id}.xml'
    fstr = os.path.join(anno,fstr)
    file = open(fstr,'wt')
    file.write(ret_str)
    s_str = f"\t<filename>{id}.jpg</filename>\n"
    file.write(s_str)
    
    ret = {}
    for i, l in enumerate(c.values):        
        c_idx = i
        
        if i == 0:
            continue
        t = int((c_idx-1)/2)
        landmark = ID2LANDMARK[t]
        if not (landmark in ret):
            ret[landmark] = [-1,-1,-1,-1]
        xy = (c_idx-1) %2
        vmin = int(max(0,625*l - size))
        vmax = int(min(625,625*l+size))
        if xy == 0:
            ret[landmark][0] = vmin
            ret[landmark][2] = vmax
        else:
            ret[landmark][1] = vmin
            ret[landmark][3] = vmax
    for i in range(len(ret)):
        landmark = ID2LANDMARK[i]
        o_str = f'\t<object>\n\t\t<name>{landmark}</name>\n\t\t<bndbox>\n\t\t\t<xmin>{ret[landmark][0]}</xmin>\n\t\t\t<ymin>{ret[landmark][1]}</ymin>\n\t\t\t<xmax>{ret[landmark][2]}</xmax>\n\t\t\t<ymax>{ret[landmark][3]}</ymax>\n\t\t</bndbox>\n\t</object>\n'
        file.write(o_str) 
    file.write('</annotation>')
    file.close() 
    print(ret) 
    break
