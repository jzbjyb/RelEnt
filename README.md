## Resources

- https://www.wikidata.org/wiki/Wikidata:Property_navboxes
- https://tools.wmflabs.org/hay/propbrowse/
- https://www.npmjs.com/package/wikidata-taxonomy
    - get subproperty: `wdtaxonomy P361`
- https://tools.wmflabs.org/prop-explorer/
    - A tree structure constructed by "subclass of" (P279) or "subproperty of" (P1647).
- https://github.com/lucaswerkmeister/wikidata-ontology-exploreradd 

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

1. download truthy file from https://dumps.wikimedia.org/wikidatawiki/entities/
2. generate triples.txt
3. downsampling
    - downsample triples to keep only frequent entities and save it to triples_ds.txt
    - downsample triples by properties
4. split leaf properties to ultra-fine properties (based on the ontology inferred from the entire triples.txt)
5. modify triples_ds.txt to reflect the changes in property hierarchy
6. train KGE methods using triples_ds.txt
