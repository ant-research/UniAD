import pandas as pd
import json

x = {'imdbid': [], 'text_gt': [], 'text_gen': [], 'char_list': []}

with open('') as f: # gt json
    gt_ads = json.load(f)

with open('') as f: # prediction ads
    generate_ads = json.load(f)

# gt_ads = gt_ads['tvad']
# generate_ads = generate_ads['tvad'] # for TVAD

gt_ads = gt_ads['cmdad']
generate_ads = generate_ads['cmdad'] # for CMDAD

# for CMDAD; to evaluate on TVAD, change corresponding files to tv*.json
with open('cmdad_charbank.json') as f:
    tt2char = json.load(f)

with open('CMD_ttmapping.json') as f:
    ttmapping = json.load(f)

with open('cmdad_chars_mapping.json') as f:
    rolename = json.load(f)

for i, ad in enumerate(gt_ads):
    try:
        imdbid = ttmapping[ad]
        char_list_ = tt2char[imdbid]
    except KeyError as e:
        continue
    char_list = []
    if len(char_list_) > 10:
        char_list_ = char_list_[:10]
    for char in char_list_:
        idc = char['id']
        try:
            char_list.append(rolename[idc]['char_in_ad'])
        except KeyError as e:
            continue
    x['imdbid'].append(imdbid)
    x['text_gen'].append(generate_ads[i])
    x['text_gt'].append(ad)
    x['char_list'].append(char_list)

dataframe = pd.DataFrame(x)
dataframe.to_csv("",index=False,sep=',') # output csv for llm_eval
