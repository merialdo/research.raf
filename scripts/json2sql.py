import psycopg2.extras
import psycopg2

from adapter import abstract_specifications_adapter
from utils import string_utils
import json
from config.bdsa_config import _config_
import os

_INSERT_SITE = "INSERT INTO site (name) VALUES %s"
_INSERT_SOURCE = "INSERT INTO source (site,category) VALUES (%s, %s);"
_INSERT_PAGE = "INSERT INTO page (site,category,url) VALUES (%s,%s,%s);"
_INSERT_ATTRIBUTE = "INSERT INTO attribute " \
                   "(page_url,key,value, key_normalized, value_normalized) VALUES %s;"
_INSERT_PID = "INSERT INTO pid (id) VALUES (%s) ON CONFLICT DO NOTHING;"
_INSERT_ID_CATEGORY = "INSERT INTO id_category (id, category) VALUES (%s,%s)"
_INSERT_ID_COMMUNITY = "INSERT INTO id_community (id, community) VALUES (%s,%s)"

_INSERT_PAGE_ID_COMMUNITY = "INSERT INTO page_id_community(url, id, community) SELECT val.url, val.id, val.community" \
 " FROM  (  VALUES %s ) val (url, id, community) JOIN page ON page.url = val.url"
_INSERT_PAGE_ID_CATEGORY = "INSERT INTO page_id_category(url, id, category) SELECT val.url, val.id, val.category" \
 " FROM  (  VALUES %s ) val (url, id, category) JOIN page ON page.url = val.url"

_MISSING_PAGE_ID = "SELECT val.url, val.id, val.category" \
 " FROM  (  VALUES %s ) val (url, id, category) LEFT JOIN page ON page.url = val.url WHERE page.url IS NULL"

def connect_db():
    """
    Connect to postgres database
    :return: 
    """
    conn = psycopg2.connect(_config_.get_option('local','connection_string'))
    return conn

def select_all_dbs(conn):
    """
    Shows all DBs in the datasource
    :param conn: 
    :return: 
    """
    cur = conn.cursor()
    cur.execute("SELECT datname from pg_database")
    rows = cur.fetchall()
    print("\nShow me the databases:\n")
    for row in rows:
        print("   ", row[0])
    cur.close()


def _insert_specifications(conn, truncate = False):
    """
    Import sources with all specifications in DB
    :param truncate: 
    :param conn: 
    :return: 
    """
    cur = conn.cursor()
    if truncate:
        _clear_database(True, False)

    all_sites = [(site,) for site in os.listdir(_config_.get_specifications())]
    psycopg2.extras.execute_values(cur, _INSERT_SITE, all_sites)

    urls = []
    for source in abstract_specifications_adapter.specifications_generator(False, False):
        cur.execute(_INSERT_SOURCE, (source.site, source.category))
        for url, specs in source.pages.items():
            if url in urls:
                print('duplicate URL: '+ url)
            else:
                urls.append(url)
                cur.execute(_INSERT_PAGE, (source.site, source.category, url))
                all_attributes = {}
                for key, value in specs.items():
                    # postgres has a limit of 2700 chars. Having this limit we cannot anyway save the entire data in any
                    # # case, so better to cut more chars --> 500
                    key = key[:500]
                    value = value[:500]
                    #use a dictionary in order to avoid double keys
                    all_attributes[key] = (url, key, value,
                          string_utils.folding_using_regex(key), string_utils.folding_using_regex(value))
                psycopg2.extras.execute_values(cur, _INSERT_ATTRIBUTE, list(all_attributes.values()))
    conn.commit()

def _insert_all_linkages(conn, truncate = False):
    """
    Import all linkages (both id--> cat --> url and id --> comm --> url)
    :param conn: 
    :param truncate: if true, truncates all existing linkages and r
    :return: 
    """
    cur = conn.cursor()
    if truncate:
        _clear_database(False, True)

    print("category linkage")
    unknown_cat = _insert_linkage(conn, _config_.get_linkage_dexter(),
                                  _INSERT_ID_CATEGORY, _INSERT_PAGE_ID_CATEGORY)
    conn.commit()
    with open(_config_.get_output_dir() + '/missing_urls_id2cat_'
              + string_utils.timestamp_string_format() +'.json', 'w') as outfile:
        json.dump(unknown_cat, outfile, indent=2)

    print("community linkage")
    unknown_comm = _insert_linkage(conn, _config_.get_community_linkage_dexter(),
                                   _INSERT_ID_COMMUNITY, _INSERT_PAGE_ID_COMMUNITY)
    with open(_config_.get_output_dir() + '/missing_urls_id2comm_'
              + string_utils.timestamp_string_format() +'.json', 'w') as outfile:
        json.dump(unknown_comm, outfile, indent=2)
    conn.commit()

def _clear_database(conn, truncate_specifications = True, truncate_linkages = True):
    cur = conn.cursor()

    if truncate_specifications:
        cur.execute('TRUNCATE site CASCADE')
        cur.execute('TRUNCATE source CASCADE')
        cur.execute('TRUNCATE page CASCADE')
        cur.execute('TRUNCATE attribute CASCADE')

    if truncate_linkages:
        cur.execute('TRUNCATE id_category CASCADE')
        cur.execute('TRUNCATE id_community CASCADE')

    conn.commit()

def insert_test_data(conn):
    cur = conn.cursor()
    ins = "INSERT INTO test_table (id, test_string) VALUES (%s,%s)"
    cur.execute(ins, ("test_id", "test_category"))

def _insert_linkage(conn, filename, insert_id2cat, insert_id2page2cat):
    """
    Generic method to build id2cat2url OR id2comm2urls
    :param conn: 
    :param linkage_file: 
    :param insert_id2cat: 
    :param insert_id2page2cat: 
    :return: 
    """

    cur = conn.cursor()
    unknown_urls = []
    with open(filename, 'r') as infile:
        id2cat2urls = json.load(infile)
        size = 0
        for id, cat2urls in id2cat2urls.items():
            if size % 1000 == 0:
                print("first " +str(size)+" ids")
            for cat, urls in cat2urls.items():
                cur.execute(insert_id2cat, (id, cat))
                # these are all data to insert, however we have to filter out entries with an unknown URL
                # i.e., a url that is not in page table
                data_to_insert = [(url, id, cat) for url in urls]
                psycopg2.extras.execute_values(cur, insert_id2page2cat, data_to_insert)

                #now retrieve missing page to put the in unkno
                #TODO too long, deleted
                # psycopg2.extras.execute_values(cur, MISSING_PAGE_ID, urls,
                #                                template="(%s, " + id + ", " + cat + ")")
                # missing = cur.fetchall()
                # unknown_urls.extend(missing)
            size = size + 1

    return unknown_urls

def insert_or_clear_data(specifications, linkage, truncate_specifications, truncate_linkage):
    conn = connect_db()
    if truncate_specifications or truncate_linkage:
        _clear_database(conn, truncate_specifications, truncate_linkage)

    if specifications:
        _insert_specifications(conn)

    if linkage:
        _insert_all_linkages(conn)

    conn.close()


def execute_methods(methods):
    """
    Create a connection, launch the list of methods with that connection then close it.
    :param methods: 
    :return: 
    """
    conn = connect_db()
    for method in methods:
        method(conn)
    conn.close()
