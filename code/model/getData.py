
import rdflib
import json

from wikidataintegrator import wdi_core
from SPARQLWrapper import SPARQLWrapper2
from SPARQLWrapper.SmartWrapper import Value

import requests
import logging
import time
from tqdm import tqdm
meta_data = None


def load_ttl(g: rdflib.Graph, datafile):
    if isinstance(datafile, list):
        for i in datafile:
            load_ttl(g, i)
    try:
        g.parse(datafile)
    except FileNotFoundError:
        print(f"FileNotFoundError when prase {datafile}")


def _query_triplet(kg, item, position=0):

    if isinstance(item, list):
        return [_query_triplet(kg, _item, position) for _item in item]
    elif isinstance(kg, rdflib.Graph):
        return query_triplet_from_graph(kg, item, position=0)
    else:
        raise NotImplementedError("will be releaased later")

def query_triplet(kg, item, position=0):
    
    try:
        res = _query_triplet(kg, item, position)
        try:
            test = res[0]
            return res
        except BaseException:
            return [[]]
    except Exception as e:
        return [[]]

def convert_item_and_pos_to_query(item, position=0):
    assert position != 1, f"item can't be relation while using graph"
    any_ls = ["?s", "?r", "?o"]
    temp = f'"{item}"'
    any_ls[position] = temp

    to_select = {
        0: " ?r ?o ",
        1: " ?s ?o ",
        2: " ?s ?r "
    }

    query = f"""select {to_select[position]} where {{ {" ".join(any_ls)} }}"""
    return query


def query_triplet_from_graph(g: rdflib.Graph, item, position=0):
    query = convert_item_and_pos_to_query(item, position)
    results = g.query(query)
    results = [
        [
            item.toPython() for item in row
        ]
        for row in results
    ]
    new_ls = [
        [
            item, i[0], i[1]
        ]
        if position == 0
        else [
            i[0], i[1], item
        ]
        for i in results
    ]
    if new_ls is None or new_ls == []:
        return [[]]
    return new_ls