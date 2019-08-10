## Resources

- https://www.wikidata.org/wiki/Wikidata:Property_navboxes
- https://tools.wmflabs.org/hay/propbrowse/
- https://www.npmjs.com/package/wikidata-taxonomy
    - get subproperty: `wdtaxonomy P361`
- https://tools.wmflabs.org/prop-explorer/
    - A tree structure constructed by "subclass of" (P279) or "subproperty of" (P1647).
- https://github.com/lucaswerkmeister/wikidata-ontology-exploreradd 
- https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all (property types)
- https://tools.wmflabs.org/hay/propbrowse/props properties in a single json file

## Build sub-graph

- [Wikidata Graph Builder](https://angryloki.github.io/wikidata-graph-builder/)
- [WikidataTreeBuilderSPARQL tutorial](https://medium.com/u-change/exploring-wikidata-for-nlp-24c4a7babf0f)
- [WikidataTreeBuilderSPARQL](https://github.com/petartodorov/WikidataTreeBuilderSPARQL)
- [Wikidata DataModel](https://www.mediawiki.org/wiki/Wikibase/DataModel/Primer)

## Implementation

- [pytorch biggraph embedding](https://ai.facebook.com/blog/open-sourcing-pytorch-biggraph-for-faster-embeddings-of-extremely-large-graphs/)
    - [Wikidata graph](https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html#wikidata)
- [gated graph nn](https://github.com/pcyin/pytorch-gated-graph-neural-network/)


## Useful SQL queries

```SQL
SELECT ?item ?itemLabel ?value ?valueLabel 
WHERE 
{
  ?item wdt:P170 ?value.  # value should be the creator of item
  #?item wdt:P136 wd:Q828322.  # item's genre must be a game
  #?item wdt:P31 wd:Q7397.  # item is an instance of software
  #?value wdt:P452 wd:Q941594.  # value's industry be a video game
  ?value wdt:P106 wd:Q5482740.  # value's occupation should be developer
  #?item ?prop ?value.
  FILTER NOT EXISTS { ?item wdt:P178 ?value }  # value is not the developer of item
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
LIMIT 100
```

## Wikidata preprocessing

1. Download truthy file from https://dumps.wikimedia.org/wikidatawiki/entities/
2. Generate triples.txt and split it by properties.
    - Only keep properties whose head and tail items are entities (starts with 'Q').
3. Downsampling
    - Downsample triples to keep only frequent entities and save it to triples_ds.txt
    - Downsample triples by properties. Keep the most popular instances for each property. 
    The number of instance kept is determined by taking sqrt or log on the size of the property.
4. Inflate the downsampled properties because a kept instance of property A might also be an instance of property B but is not select.
Make sure that all the entities kept have their P31 being kept because we use this to split properties.
5. Build an ontology (based on the entire Wikidata, not the downsampled one) for the dataset using P31 (instance of) and P279 (subclass of). 
The classification system of Wikidata is very strange, which is an cyclic graph.
6. Split leaf properties to ultra-fine properties based on the ontology inferred from above.
    1. Compute the depth of each item in the ontology. Several heuristics need be used because it is cyclic.
    2. For each instance of a certain property, we use the value of P31 of its head entity and tail entity as signature to split.
    In cases where the entities don't have P31, we use 'Q' as the placeholder.
    3. For a property with K instances, all its sub-properties larger than K/100 are kept and the remaining ones are merged if any, 
    which mean that the maximum number of sub-properties we could get is 100+1.
7. Merge instances from all the properties generated by the splitting algorithm.
8. Train KGE methods using the merged file.
9. Choose new parent among the sub-properties. This is crucial because it will influence the perform significantly.

## [SLING Python API](https://github.com/google/sling/blob/master/doc/guide/pyapi.md)

```python
# get string of the mention
mention_str = doc.phrase(mention.begin, mention.end)

# iterate over evokes
for e in mention.evokes():
    print(e.data(pretty=True))
```

## Experiments

[Google sheet](https://docs.google.com/spreadsheets/d/17jdKww8Ao6B8ahfjQMhQ3dzlBWdc1zAHSxcU6yz8nf8/edit?usp=sharing) used to track experimental resutls.