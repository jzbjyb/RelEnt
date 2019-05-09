from SPARQLWrapper import SPARQLWrapper, JSON


ENDPOINT = "https://query.wikidata.org/sparql"


# two slots: property and limit
PROPERTY_QUERY = """#All items with a property
# Sample to query all values of a property
# Property talk pages on Wikidata include basic queries adapted to each property
SELECT
  ?item ?itemLabel
  ?value ?valueLabel
# valueLabel is only useful for properties with item-datatype
  (MD5(CONCAT(str(?item), str(?value), str(RAND()))) as ?random)
WHERE 
{
  ?item wdt:%s ?value
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?random
# remove or change limit for more results
LIMIT %d"""

# two slots: property and limit
PROPERTY_QUERY_FAST = """#All items with a property
# Sample to query all values of a property
# Property talk pages on Wikidata include basic queries adapted to each property
SELECT
  ?item ?itemLabel
  ?value ?valueLabel
# valueLabel is only useful for properties with item-datatype
WHERE 
{
  ?item wdt:%s ?value
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
# remove or change limit for more results
LIMIT %d"""


def get_results(endpoint_url, query, timeout=None):
    sparql = SPARQLWrapper(endpoint_url)
    if timeout:
        sparql.setTimeout(timeout)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_property_occurrence(pid, limit=1000, timeout=None, fast=False):
    if fast:
        q = PROPERTY_QUERY_FAST % (pid, limit)
    else:
        q = PROPERTY_QUERY % (pid, limit)
    return get_results(ENDPOINT, q, timeout=timeout)
