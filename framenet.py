
import xml.etree.ElementTree as ET


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



def dicesim(A, B):
    A = set(A)
    B = set(B)
    return (( 2 * len(A & B) ) / ( len(A) + len(B) ))


# assume a dict input {frame: list of verbs}
# finds model frame with best dice score for each FN frame
def dicematch(model_frameverbs, fn_frameverbs):
    
    # dict {FN frameID: {model frame: dicesim score with FN frame}}
    frame_matches = dict.fromkeys(fn_frameverbs)
    
    for fn_frame in fn_frameverbs:
        dicesims = dict.fromkeys(model_frameverbs)
        for m_frame in model_frameverbs:
            dicesims[m_frame] = dicesim(fn_frameverbs[fn_frame], model_frameverbs[m_frame])
        frame_matches[fn_frame] = dicesims

    return frame_matches


def dicemax(model_frameverbs, fn_frameverbs):
    dicedict = dicematch(model_frameverbs, fn_frameverbs)

    frame_maxes = dict.fromkeys(fn_frameverbs)

    for fn_frame in fn_frameverbs:
        currentmax = ('',-1)
        for m_frame in model_frameverbs:
            if dicedict[fn_frame][m_frame] > currentmax[1]:
                currentmax = (m_frame, dicedict[fn_frame][m_frame])
        frame_maxes[fn_frame] = currentmax

    return frame_maxes



#def get_model_verbsets(frame_assign):
