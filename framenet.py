
import xml.etree.ElementTree as ET


##------------

lutree = ET.parse('fndata-1.5/luIndex.xml')
luroot = lutree.getroot()
luchildren = [i for i in luroot]
ludicts = [i.attrib for i in luchildren[1:]]

frametree = ET.parse('fndata-1.5/frameIndex.xml')
frameroot = frametree.getroot()
framechildren = [i for i in frameroot]
framedicts = [i.attrib for i in framechildren]

fnframes = {i['ID']: i['name'] for i in framedicts}

luverbdicts = []
for i in ludicts:
    if i['name'][-2:] == '.v':
        luverbdicts.append(i)



def print_lu_verbs():
    for i in luverbdicts:
        print(i['ID'].rjust(6),i['name'][:-2].center(21),'|',i['frameID'].rjust(6),i['frameName'])

def remove_sparse_frames(d, thresh=0):
    d2 = d.copy()
    for i in d2:
        if len(d2[i]) <= thresh:
            del d[i]

# returns {frameID: [list of verbs]} dict
def sort_verbs_to_frames(thresh=0):
    frameverbs = dict.fromkeys([i['ID'] for i in framedicts], [])
    for fID in frameverbs.keys():
        for v in luverbdicts:
            if v['frameID'] == fID:
                # remove multi-word verbs
                if (' ' not in v['name']) and ('_' not in v['name']):
                    frameverbs[fID] = frameverbs[fID] + [v['name'][:-2]]
    # remove frames that have fewer than thresh assocatied verbs
    remove_sparse_frames(frameverbs, thresh)
    return frameverbs

