import pandas as pd
import csv
import os

# Funzione che prende una URL e il json di "matching" delle entity e restituisce la entity corrispondente alla URL se presente, NO_<ID_URL> altrimenti
from external_tools.t2k import t2k_utils, t2k_adapted
from external_tools.t2k.t2k_utils import import_json_source, get_entity_label
from utils import bdsa_utils

URL = 'url'


def get_all_json_attributes(camera_spec_data):
    """
    Funzione che estrapola il nome di tutti gli attributi presenti in tutto il json, senza duplicati
    """
    attributes = {URL, t2k_adapted.LABEL_KB}
    attributes.update({att for camera_spec in camera_spec_data for att in camera_spec['spec'].keys()})
    return attributes

# Funzione che genera le prime 3 righe particolari
def generate_additional_header_rows(attributes, source_name):
    """
    Return first three additional header rows
    """
    attributes_to_add = attributes - {'url', t2k_adapted.LABEL_KB}
    rows = [
        { 
            'url': "URI", 
            'spec': {**{t2k_adapted.LABEL_KB: t2k_adapted.LABEL_KB_FULL}, **{
                name: t2k_utils.rename_column(name, source_name) for name in attributes_to_add
            }}
        }, 
        { 
            'url': "URI", 
            'spec': {**{t2k_adapted.LABEL_KB: t2k_adapted.LITERAL_SHORT}, **{
                name: "XMLSchema#string" for name in attributes_to_add
            }}
        },
        { 
            'url': "http://www.w3.org/2002/07/owl#Thing", 
            'spec': {**{t2k_adapted.LABEL_KB: t2k_adapted.LITERAL_LONG}, **{
                name: "http://www.w3.org/2001/XMLSchema#string" for name in attributes_to_add
            }}
        }
    ]
    return rows

def produce_output(source_directory, source_name, category,output_dir):
    """
    Funzione principale che si occupa di richiamare tutte le altre
    """
#    print('Script started')
    # Caricamento file csv 'camera_spec.json' e 'camera_linkage.json'
#   print('Opening required files')
    camera_linkage_data, camera_spec_data = import_json_source(category, source_directory)
    #    print('Opened linkage and spec files')

    camera_spec_data = generate_additional_header_rows(get_all_json_attributes(camera_spec_data), source_name) + camera_spec_data

    # Per ogni oggetto nel json viene cambiato il valore del campo '<page title> con la entity presente nel json 'camera_linkage.json' e se non è presente
    # Viente sostituito con una stringa NO_<ID_URL>
#    print("Replacing all <page title>\'s values")
    url2linkage = bdsa_utils.multidict_inverter(camera_linkage_data)
    for camera_spec in camera_spec_data:
        current_url = camera_spec['url']
        if 'URI' not in current_url and 'http://www.w3.org/2002/07/owl#Thing' not in current_url:
            first_label = get_entity_label(current_url, url2linkage)
            camera_spec['spec'][t2k_adapted.LABEL_KB] = first_label

    # trasformarmazione del Json in CSV
#   print('Normalizing json')
    knowledge_base_output = pd.json_normalize(camera_spec_data)
#   print("Json normalized... Producing CSV")
    knowledge_base_output.drop_duplicates(inplace=True, subset=set(knowledge_base_output.columns) - {'url'})

    #some normalization
    knowledge_base_output.rename(inplace=True, columns=lambda x: x.replace('spec.', ''))
    knowledge_base_output.rename(inplace=True, columns={'url': 'URI'})
    knowledge_base_output.to_csv(os.path.join(output_dir, 'dbpedia.csv'), sep=',', index=None, na_rep="", quotechar='"', quoting=csv.QUOTE_ALL,
                           escapechar='\\')
    print('KB produced')
    return knowledge_base_output