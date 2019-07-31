import logging
from typing import NamedTuple, Tuple, List, Iterable
from pathlib import Path
from tqdm import tqdm
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


def load(record: str) -> Iterable[Tuple[sling.nlp.document.Document, Tuple[int, str, str]]]:
    """load documents from a .rec file.
    Warning: this may take good amount of RAM space (each *.rec file is 5.3GB).
    """
    for k, rec in tqdm(sling.RecordReader(record)):
        store = sling.Store(commons)
        # parse record into frame
        doc_frame = store.parse(rec)
        # instantiate a document
        parsed_doc = sling.Document(doc_frame, store, DOCSCHEMA)
        metadata = get_metadata(doc_frame)
        yield parsed_doc, metadata


def get_mentions(document: sling.nlp.document.Document) -> List[Tuple[int, int, str]]:
    """ Returns the string ID of the linked entity for this mention.
    Credit: Thanks Bhuwan for sharing the code.
    """
    mentions = document.mentions
    linked_mentions: List[Tuple[int, int, str]] = []
    for i, mention in enumerate(mentions):
        # get position
        start, end = mention.begin, mention.end
        # get wikidata id
        if "evokes" not in mention.frame or type(mention.frame["evokes"]) != sling.Frame:
            continue
        if "is" in mention.frame["evokes"]:
            if type(mention.frame["evokes"]["is"]) != sling.Frame:
                if "isa" in mention.frame["evokes"] and \
                        mention.frame["evokes"]["isa"].id == "/w/time" and \
                        type(mention.frame["evokes"]["is"]) == int:
                    linked_mentions.append((start, end, mention.frame["evokes"]["is"]))
                else:
                    continue
            else:
                linked_mentions.append((start, end, mention.frame["evokes"]["is"].id))
        else:
            linked_mentions.append((start, end, mention.frame["evokes"].id))
    return linked_mentions


if __name__ == "__main__":
    record_files = ANNOTATED_DIR.glob("*.rec")
    
    for rec_file in record_files:
        for doc, metadata in load(str(rec_file)):
            mentions = get_mentions(doc)
        break  # TODO: Remove this line.
