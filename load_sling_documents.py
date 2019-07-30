import logging
from typing import NamedTuple
from pathlib import Path

import sling

LOGGER = logging.getLogger(__name__)
ANNOTATED_DIR = Path("/home/hiroakih/tir3/sling/local/data/e/ner/en")

# some magic commands
commons = sling.Store()
DOCSCHEMA = sling.DocumentSchema(commons)
commons.freeze()


def get_metadata(frame: sling.Frame) -> Tuple[int, str, str]:
    """retrieves article information other than text.
    See https://github.com/google/sling/blob/master/doc/guide/wikiflow.md#wikipedia-import-and-parsing 
    for more information about what can be extracted (e.g., category, theme)
    """
    pageid = frame.get("/wp/page/pageid")  # wikiPEDIA page ID
    title = frame.get("/wp/page/title")  # article title
    item = frame.get("/wp/page/item")  # wikidata ID associated to the article
    return pageid, title, item


def load(record: str) -> List[Tuple[sling.nlp.document.Document, Tuple[int, str, str]]]:
    """load documents from a .rec file.
    Warning: this may take good amount of RAM space (each *.rec file is 5.3GB).
    """
    doc = sling.RecordReader(record)

    result = []
    for k, rec in sling.RecordReader(record):
        store = sling.Store()
        # load a record by mapping the content into document schema
        parsed_doc = sling.Document(store.parse(rec), store, DOCSCHEMA)
        # parse again, without the schema
        # TODO: can we just parse once?
        metadata = get_metadata(store.parse(rec))  
        result.append((parsed_doc, metadata))
    
    return result


def get_mentions(document: sling.nlp.document.Document):
    """ Returns the string ID of the linked entity for this mention.
    Credit: Thanks Bhuwan for sharing the code.
    """
    mentions = document.mentions
    linked_mentions = []
    for i, mention in enumerate(mentions):
        if "evokes" not in mention.frame or type(mention.frame["evokes"]) != sling.Frame:
            continue

        if "is" in mention.frame["evokes"]:
            if type(mention.frame["evokes"]["is"]) != sling.Frame:
                if ("isa" in mention.frame["evokes"] and 
                    mention.frame["evokes"]["isa"].id == "/w/time" and 
                    type(mention.frame["evokes"]["is"]) == int):
                    linked_mentions.append((i, mention.frame["evokes"]["is"]))
                else:
                    continue
            else:
                linked_mentions.append((i, mention.frame["evokes"]["is"].id))
        linked_mentions.append((i, mention.frame["evokes"].id))
    return linked_mentions


if __name__ == "__main__":
    record_files = ANNOTATED_DIR.glob("*.rec")
    
    for rec_file in record_files:
        articles = load(rec_file)
        for article in articles:
            mentions = get_mentions(article)

        break  # TODO: Remove this line.




