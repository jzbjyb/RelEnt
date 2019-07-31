import logging
from typing import NamedTuple, Tuple, List, Iterable
from pathlib import Path
from tqdm import tqdm
import sling
from sling.nlp.document import DocumentSchema, Token, Mention

LOGGER = logging.getLogger(__name__)
ANNOTATED_DIR = Path("/home/hiroakih/tir3/sling/local/data/e/ner/en")

# some magic commands
commons = sling.Store()
DOCSCHEMA = sling.DocumentSchema(commons)
commons.freeze()


class MyDocument(object):
    def __init__(self, frame=None, store=None, schema=None, load_tokens=True):
        # Create store, frame, and schema if missing.
        if frame != None:
            store = frame.store()
        if store == None:
            store = sling.Store()
        if schema == None:
            schema = DocumentSchema(store)
        if frame == None:
            frame = store.frame([(schema.isa, schema.document)])

        # Initialize document from frame.
        self.frame = frame
        self.schema = schema
        self._text = frame.get(schema.document_text, binary=True)
        self.tokens = []
        self.mentions = []
        self.themes = []
        self.tokens_dirty = False
        self.mentions_dirty = False
        self.themes_dirty = False

        if load_tokens:  # Get tokens.
            tokens = frame[schema.document_tokens]
            if tokens != None:
                for t in tokens:
                    token = self.get_word(t, schema, self._text)
                    self.tokens.append(token)

        # Get mentions.
        for m in frame(schema.document_mention):
            mention = Mention(schema, m)
            self.mentions.append(mention)


    def get_word(self, frame, schema, _text):
        text = frame[schema.token_word]
        if text == None:
            start = frame[schema.token_start]
            if start != None:
                size = frame[schema.token_size]
                if size == None: size = 1
                text = _text[start: start + size].decode()
        return text


def get_metadata(frame: sling.Frame) -> Tuple[int, str, str]:
    """retrieves article information other than text.
    See https://github.com/google/sling/blob/master/doc/guide/wikiflow.md#wikipedia-import-and-parsing 
    for more information about what can be extracted (e.g., category, theme)
    """
    pageid = frame.get("/wp/page/pageid")  # wikiPEDIA page ID
    title = frame.get("/wp/page/title")  # article title
    item = frame.get("/wp/page/item")  # wikidata ID associated to the article
    return pageid, title, item


def load(record: str, load_tokens: bool = True) -> Iterable[Tuple[sling.nlp.document.Document, Tuple[int, str, str]]]:
    """load documents from a .rec file.
    Warning: this may take good amount of RAM space (each *.rec file is 5.3GB).
    """
    for k, rec in sling.RecordReader(record):
        store = sling.Store(commons)
        # parse record into frame
        doc_frame = store.parse(rec)
        # instantiate a document
        #parsed_doc = sling.Document(doc_frame, store, DOCSCHEMA)
        parsed_doc = MyDocument(doc_frame, store, DOCSCHEMA, load_tokens=load_tokens)
        metadata = get_metadata(doc_frame)
        yield parsed_doc, metadata


def get_mentions(document: sling.nlp.document.Document) -> Iterable[Tuple[int, int, str]]:
    """ Returns the string ID of the linked entity for this mention.
    Credit: Thanks Bhuwan for sharing the code.
    """
    for i, mention in enumerate(document.mentions):
        # get position
        start, end = mention.begin, mention.end
        # get wikidata id
        if "evokes" not in mention.frame or type(mention.frame["evokes"]) != sling.Frame:
            continue
        if "is" in mention.frame["evokes"]:
            if type(mention.frame["evokes"]["is"]) != sling.Frame:
                continue
                '''
                if "isa" in mention.frame["evokes"] and \
                        mention.frame["evokes"]["isa"].id == "/w/time" and \
                        type(mention.frame["evokes"]["is"]) == int:
                    yield start, end, mention.frame["evokes"]["is"]
                else:
                    continue
                '''
            else:
                yield start, end, mention.frame["evokes"]["is"].id
        else:
            yield start, end, mention.frame["evokes"].id


if __name__ == "__main__":
    record_files = ANNOTATED_DIR.glob("*.rec")
    
    for rec_file in record_files:
        for doc, metadata in load(str(rec_file)):
            mentions = get_mentions(doc)
        break  # TODO: Remove this line.
