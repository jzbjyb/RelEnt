## Resources

- https://www.wikidata.org/wiki/Wikidata:Property_navboxes
- https://tools.wmflabs.org/hay/propbrowse/
- https://www.npmjs.com/package/wikidata-taxonomy
    - get subproperty: `wdtaxonomy P361`
- https://tools.wmflabs.org/prop-explorer/
    - A tree structure constructed by "subclass of" (P279) or "subproperty of" (P1647).
    
## Build sub-graph

- [Wikidata Graph Builder](https://angryloki.github.io/wikidata-graph-builder/)
- [WikidataTreeBuilderSPARQL tutorial](https://medium.com/u-change/exploring-wikidata-for-nlp-24c4a7babf0f)
- [WikidataTreeBuilderSPARQL](https://github.com/petartodorov/WikidataTreeBuilderSPARQL)
- [Wikidata DataModel](https://www.mediawiki.org/wiki/Wikibase/DataModel/Primer)

## Implementation

- [pytorch biggraph embedding](https://ai.facebook.com/blog/open-sourcing-pytorch-biggraph-for-faster-embeddings-of-extremely-large-graphs/)
    - [Wikidata graph](https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html#wikidata)
- [gated graph nn](https://github.com/pcyin/pytorch-gated-graph-neural-network/)