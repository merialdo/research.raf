import json
import pandas as pd
import os
import csv

from external_tools.t2k import t2k_utils, t2k_adapted
from utils import bdsa_utils

URL = 'url'

URI = 'URI'


def getIdentifierFromUrl(url):
    urlComponents = url.split("/")
    return urlComponents[len(urlComponents)-1]

def getUrl(urlToFind, camera_linkage_data, source):
    for camera_linkage in camera_linkage_data:
        for url in camera_linkage_data[camera_linkage]:
            if urlToFind == url: 
                return camera_linkage
    return "NO_%s_%s" % (source, getIdentifierFromUrl(urlToFind))
    
# Funzione che estrapola il nome di tutti i gli attributi presenti in tutto il json, senza duplicati
def get_all_json_attributes(camera_spec_data):
    attributes = ['url', '<page title>']
    for camera_spec in camera_spec_data:
        for attribute in camera_spec['spec']:
            if not attribute in attributes:
                attributes.append(attribute)
    return attributes

# Funzione che prende in input il json annidato e restituisce un json su un singolo livello
def produce_plane_json(camera_spec_data):
    child_count = 0
    attributes = get_all_json_attributes(camera_spec_data)
    #print('Producing plane json from original json in camera_spec.json')
    plane_json = []
    # Per ogni oggetto nel json annidato
    for camera_spec in camera_spec_data:
        if("NO_" not in camera_spec['spec']['<page title>']):
            # Creo un json con tutti gli attributi e tutti valori a NULL
            plane_json_child = { prop_name: "NULL" for prop_name in attributes }
            # Popolo i valori esistenti nel json originario
            for attribute in camera_spec['spec']:
                try:
                    value = camera_spec['spec'][attribute]
                    plane_json_child[attribute] = value.replace("\n", "").replace("\"", "")
                except KeyError:
                    plane_json_child = "NULL"
            # Aggiungo anche la URL
            plane_json_child['url'] = camera_spec['url']
            plane_json.append(plane_json_child)
            child_count = child_count + 1
    print("Plane json produced")
    return plane_json

def generate_csv_header(attributes):
    header = ""
    for attribute in attributes:
        header = header + '"' + attribute + '",'
    return header[:-1]

def create_csv_row_from_json(json, attributes):
    row = ""
    for attribute in attributes:
        row = row + '"' + json[attribute] + '",'
    return row[:-1] + '\n'

def produceOutput(source_directory, category, output_dir, source):
    camera_linkage_data, camera_spec_data = t2k_utils.import_json_source(category, source_directory)
    url2labels = bdsa_utils.multidict_inverter(camera_linkage_data)
    for camera_spec in camera_spec_data:
        camera_spec['spec'][t2k_adapted.LABEL_WT] = t2k_utils.get_entity_label(camera_spec[URL],
                                                                               url2labels, source)

    # print('Normalizing json')
    webtable = pd.json_normalize(camera_spec_data)
    webtable.drop_duplicates(inplace=True, subset=set(webtable.columns) - {URL})
    #print("Json normalized... Producing CSV")
    webtable.rename(inplace=True, columns=lambda x: x.replace('spec.', ''))
    webtable.rename(inplace=True, columns={URL: URI})

    #Reorder data
    cols = set(webtable.columns.tolist())
    cols.difference_update({URI, t2k_adapted.LABEL_WT})
    webtable = webtable[[URI, t2k_adapted.LABEL_WT] + list(cols)]
    webtable.to_csv(os.path.join(output_dir, 'webtable.csv'), sep=',', index=None, na_rep="", quotechar='"', quoting=csv.QUOTE_ALL)
    print('Webtable CSV produced')
    return webtable
    
    # fin = open("../../../../../../projects/t2k/webtables_generator/webtable.csv", "rt")
    # data = fin.read()
    # data = data.replace('spec.', '')
    # data = data.replace("<page title>", 'label_name')
    # fin = open("../../../../../../projects/t2k/webtables_generator/webtable.csv", "wt")
    # fin.write(data)
    # fin.close()
