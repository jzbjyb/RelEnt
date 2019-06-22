import argparse
import json
import logging
import pickle
import re
from collections import deque, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, NamedTuple
from itertools import chain
from pathlib import Path

import tqdm
import numpy as np
from nltk.corpus import stopwords


LOGGER = logging.getLogger(__name__)
STOPWORDS = set(stopwords.words("english") + [".", ",", '"'])
PAREN = re.compile(r"([^\(\s]+)\s\([^\)]+\)")  # Parenthesis removal


class Node(NamedTuple):
    id: str
    relations: List[str]
    depth: int
    head_id: str  # Head entity where the relation comes from.


class WikidataGraph:

    """A class object containing the full graph information."""

    def __init__(self, id2name, id2nb, id2date):
        self.id2name = id2name
        self.id2date = id2date
        self.id2nb = id2nb

    def expand_graph(self, query: Node, distance: int = 1) -> List[Node]:
        """given an ID, get a neighbor with BFS"""
        assert distance > 0
        depth_limit = query.depth + distance
        retrieved = [Node(n, [rel], query.depth+1, query.id) for rel, n in self.id2nb.get(query.id, [])]

        if distance == 1:
            return retrieved
        else:
            q = deque(retrieved)
            while len(q) > 0:
                node = q.popleft()
                if node.depth > depth_limit:
                    break

                retrieved.append(node)
                for r, n in self.id2nb.get(node.id, []):
                    q.append(Node(n, node.relations + [r], node.depth + 1, node.id))

        return retrieved

    def expand_graph_me(self, query: Node, distance: int = 1) -> List[Node]:
        """given an ID, get a neighbor with BFS"""
        assert distance > 0
        depth_limit = query.depth + distance
        retrieved = [Node(n, [rel], query.depth+1, query.id) for rel, n in self.id2nb.get(query.id, [])]

        if distance == 1:
            return retrieved
        else:
            q = deque(retrieved)
            while len(q) > 0:
                node = q.popleft()
                if node.depth >= depth_limit:
                    break

                for r, n in self.id2nb.get(node.id, []):
                    nn = Node(n, node.relations + [r], node.depth + 1, node.id)
                    retrieved.append(nn)
                    q.append(nn)

        return retrieved

    def get_surface_forms(self, nodes: List[Node]) -> List[Tuple[str, str, List[str], List[str], int]]:
        """get surface forms of the given list of nodes.

        returns (ID, head ID, surface forms, relation from the previous entity, depth from the root entity)
        """

        surface_forms = []
        for node in nodes:
            expressions = self.id2name.get(node.id)
            if expressions is not None:
                canonical = re.sub(PAREN, r"\1", expressions[0])
                aliases = list(set([re.sub(PAREN, r"\1", e.strip()) for e in expressions[1:]]))  # remove duplicates
                surface_forms.append((node.id, node.head_id, [canonical]+aliases, node.relations, node.depth))

        return surface_forms


class TopicGraph:

    """A subgraph per document."""

    def __init__(self, title_entity: Node, canonical: str):
        self.root = title_entity
        self.root_canonical = canonical
        self.heads = [title_entity]
        self.entities = {}  # id -> entity
        self.surfaces = {}  # id -> List[surface]

    def expand(self, query: str, kg: WikidataGraph):

        if query is None:  # Search the graph from the title entity if query is not given.
            query = self.root.id
            query_node = self.root
        elif query in self.entities:
            query_node = self.entities[query]
        else:
            # Nothing to expand.
            return

        self.heads.append(query)
        retrieved = kg.expand_graph(query_node, distance=1)
        for r in retrieved:
            self.entities[r.id] = r
        self.surfaces[query] = kg.get_surface_forms(retrieved)

    def add_surface_forms(self, node_id: str, tokens: List[Tuple[str, str, List[str], List[str], int]]):
        """Add surface forms to currently-focused entity."""
        self.surfaces[node_id] = self.surfaces[node_id] + tokens


def load_corpus(wikitext_path: Path):
    """Load the dataset from wikitext directory, and concatenate sections into single string."""
    dataset = {}
    skipped = {}
    for mode in ["train", "valid", "test"]:
        with (wikitext_path / f"{mode}.json").open("r") as f:
            data = json.load(f)
        skipped[mode] = [d for d in data if d["id"] == ""]

        data = [d for d in data if len(d["id"]) > 0]
        for i, article in enumerate(data):
            # Join section text into single string
            full_body = f"= {article['title'].strip()} = "
            for head, sec in article["sections"]:
                head = head.strip()
                if head != "":
                    head = f"= = {head.strip()} = = "
                sec = sec.strip() + " "
                if sec.strip() == "":
                    sec = ""
                full_body += head + re.sub(r"\s+", " ", sec)

            data[i]["tokens"] = full_body

        dataset[mode] = data

    return dataset, skipped


def find_matches(article,
                 title_canonical_form: str,
                 kg: WikidataGraph,
                 match_aka: bool = False,
                 match_date: bool = True,
                 match_title: bool = True,
                 match_lower: bool = True,
                 match_stopwords: bool = False):

    def remove_duplicate_occurrence(matches: List[Tuple[str, str, Tuple[int, int], List[str], str, List[str]]]) -> Dict[str, int]:
        res = {}
        for m in matches:
            if m[0] != "" and (m[0] not in res or m[2][0] < res[m[0]][0]):
                res[m[0]] = m[2]
        return res

    title, title_id, full_article_body = article["title"], article["id"], article["tokens"]

    # prepare a graph.
    graph = TopicGraph(Node(title_id, [], 0, ""), title_canonical_form)

    entities = {}
    checked_entities = []

    if match_lower:
        full_article_body = full_article_body.lower()
    tokens = full_article_body.strip().split(" ")

    tok_array = np.array(tokens)
    is_entity = np.zeros_like(tok_array, dtype=np.int)

    indicator = np.ones_like(is_entity, dtype=np.int)
    # Remove stopword counts from denominator array
    if not match_stopwords:
        for sw in STOPWORDS:
            indicator[np.where(tok_array == sw)] = 0
    n_tokens = len(tok_array)

    str_idx = 0  # char index for the article string.
    tok_idx = 0  # tok index for the list of tokens.
    all_matches = []
    not_founds = []
    ent_id = graph.heads[0].id  # First entity to check.
    # Expand and get the new surfaces.
    graph.expand(None, kg)
    surfaces = graph.surfaces[ent_id]

    if match_title:
        # ALWAYS put canonical forms at the first index of the surface forms.
        if title_canonical_form == title:
            surfaces += ([("", ent_id, [title], ["@TITLE@"], 1)])
        else:
            surfaces += ([("", ent_id, [title_canonical_form, title], ["@TITLE@"], 1)])

    while tok_idx < n_tokens:

        if match_date:
            if ent_id in kg.id2date and kg.id2date[ent_id][1] == 0:
                dates = kg.id2date.get(ent_id)[0]
                kg.id2date[ent_id][1] = 1
            else:
                dates = []

            surfaces += [("", ent_id, d[0], d[1], d[2]) for d in dates]

        sub_is_entity, found, notfound = find_single_entity_matches(
            full_article_body[str_idx:], tok_array[tok_idx:], names=surfaces,
            offset=tok_idx,
            match_aka=match_aka, match_lower=match_lower,
            match_stopwords=match_stopwords
        )
        all_matches += found
        not_founds += notfound
        is_entity[tok_idx:] = np.maximum(is_entity[tok_idx:], sub_is_entity)
        checked_entities.append(ent_id)
        if ent_id in entities:
            del entities[ent_id]

        entities.update(remove_duplicate_occurrence(found))

        ent_updated = False
        for e, (begin, end) in sorted(entities.items(), key=lambda x: x[1]):
            if e not in checked_entities and e != "":
                ent_id, tok_idx = e, end
                ent_updated = True
                break

        # No more un-checked entities left.
        if not ent_updated:
            break

        str_idx = len(" ".join(list(tok_array[:tok_idx]))) + 1

        graph.expand(ent_id, kg)
        surfaces = graph.surfaces[ent_id]

    return is_entity, all_matches, not_founds


def find_single_entity_matches(
    article: str,
    tok_array: np.ndarray,
    names: List[Tuple[str, str, List[str], List[str], int]],
    offset: int = 0,
    match_aka: bool = False,
    match_lower: bool = True,
    match_stopwords: bool = False,
):
    """TODO

    :param article: Article text in a single string.
    :param tok_array:
    :param names:
    :param match_aka: A flag to include aliases.
    :param match_lower: A flag to include lowercased matches.
    :param match_stopwords: A flag to include stopwords.
    :return: TODO
    """
    is_entity = np.zeros_like(tok_array, dtype=np.int)
    indicator = np.ones_like(is_entity, dtype=np.int)
    not_found = []
    found_ents = []

    for ent_id, head_id, surfaces, rels, _ in names:
        if not match_aka:
            # Keep only the canonical form
            surfaces = surfaces[:1]

        for surface in surfaces:
            if len(surface.strip()) == 0:
                continue

            # single word entity
            if len(surface.split(" ")) == 1:
                if not match_stopwords and surface in STOPWORDS:
                    continue

                if match_lower:
                    surface = surface.lower()

                match_idx = np.where(tok_array == surface)
                if match_idx[0].size == 0:  # Couldn't be found.
                    not_found.append((ent_id, head_id, (-1, -1), rels, surface, surfaces))
                else:
                    is_entity[match_idx] = 1
                    found_ents += [(ent_id, head_id, (i, i + 1), rels, surface, surfaces) for i in match_idx[0]]

            # multi-word entity: concatenate it and match, and then split the string back
            else:
                # multi-word entity matching
                concat_surface = surface.replace(" ", "####")
                if match_lower:
                    concat_surface = concat_surface.lower()

                try:
                    # Insert concat chars in the original article, and split again if there's any merged.
                    subbed_article, cnt = re.subn(re.escape(surface), concat_surface, article)
                    if cnt > 0:
                        phrase_len = len(surface.split(" "))
                        subbed_tok_array = np.array(subbed_article.split(" "))
                        match_idx = np.where(subbed_tok_array == concat_surface)[0]
                        match_tok_idx = [
                            range(pos + (phrase_len - 1) * i, pos + (phrase_len - 1) * i + phrase_len)
                            for i, pos in enumerate(match_idx)
                        ]
                        is_entity[np.array(list(chain.from_iterable(match_tok_idx)), dtype=np.int)] = 1
                        found_ents += [(ent_id, head_id, (i.start, i.stop), rels, surface, surfaces) for i in match_tok_idx]

                    else:
                        not_found.append((ent_id, head_id, (-1, -1), rels, surface, surfaces))

                except Exception:  # TODO: more specific exception handling.
                    LOGGER.info(f"Bad name : {surface}")
                    not_found.append((ent_id, head_id, (-1, -1), rels, surface, surfaces))

    found_ents = sorted(found_ents, key=lambda x: x[2][0])  # sort by earliest appearances.
    found_ents = [(i, j, (k[0]+offset, k[1]+offset), l, m, n)
                  for (i, j, k, l, m, n) in found_ents]

    return is_entity, found_ents, not_found


def parse_articles(
    dataset,
    canonicals,
    kg,
    match_aka=False,
    match_title=True,
    match_date=True,
    match_lower=False,
    match_stopwords=False,
):
    """annotates entities/objects of the topic entity in the respective articles.

    :param dataset:
    :param canonicals:
    :param kg:
    :param match_aka:
    :param match_title:
    :param match_date:
    :return:
    """
    stats = []
    skipped = []

    for idx, data in tqdm.tqdm(enumerate(dataset), ncols=80, desc="Searching over dataset", ascii=True):
        canonical_form = canonicals[idx]
        matches = find_matches(data, canonical_form, kg, match_aka, match_date, match_title, match_lower, match_stopwords)

        stats.append(matches)

    return stats, skipped


def recall(stats):
    """Compute macro/micro average of recall."""
    macro_vals = []
    micro_counts = np.array([0, 0])
    for is_entity, n_tokens, _ in stats:
        c = np.sum(is_entity)
        micro_counts += np.array([c, n_tokens])
        macro_vals.append(float(c) / n_tokens)
    return np.mean(macro_vals), micro_counts[0] / micro_counts[1]


def expand_date(date):
    """Get multiple surface form date expressions from a date object."""

    def convert_datetime(date_string):
        year = "%Y"
        if date_string.startswith("-"):
            year = "BC%Y"
            date_string = date_string[1:]

        fmts = [f"%-d %B {year}", f"%B %-d , {year}"]
        date_obj = datetime.strptime(date_string, "%Y-%m-%dT00:00:00Z")
        s = [datetime.strftime(date_obj, f) for f in fmts]
        return s

    abb2name = {"dob": "P569", "dod": "P570"}
    res = []
    for k, v in date.items():
        if k in ["dod", "dob"]:
            try:
                res += [[convert_datetime(v), [abb2name[k]], 1]]
            except ValueError:
                LOGGER.info(f"Date {v} not found.")
    return res


# No need for id2name, the previous preprocessing script converts the IDs
def convert_ids(data, prop2name):
    """Convert IDs into actual surface forms."""
    new_rels = []
    for d in data:
        is_entity, rels = d
        named = [
            (
                ent_id,
                head_id,
                (int(span[0]), int(span[1])),
                (rr, [prop2name.get(r, "NO_REL") for r in rr]),
                surface,
                surfaces,
            )
            for ent_id, head_id, span, rr, surface, surfaces in rels
        ]
        new_rels.append((is_entity, named))

    return new_rels


def main(args):

    wikitext_dir = Path(args.wikitext_dir)
    wikidata_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)
    
    ids = open("data/property_eid/property_occurrence_all.eid", "r").read().strip().split("\n")

    # Make the save dir if it doesn't exist
    # if not output_dir.exists():
    #     output_dir.mkdir()
    # LOGGER.info(f"Save directory is {output_dir}")

    # splits = ["train", "valid", "test"]

    # dataset, noid_dataset = load_corpus(wikitext_dir)
    # LOGGER.info("Dataset loaded.")

    # # Load canonical_forms
    # canonical_forms = {}
    # for mode in splits:
    #     with (wikitext_dir / f"{mode}_canonical_forms.txt").open("r") as f:
    #         canonical_forms[mode] = [l.strip() for l in f]
    #         assert len(canonical_forms[mode]) == len(dataset[mode])
    # LOGGER.info("Loaded canonical forms.")

    # Assuming that triples are separated in multiple files
    with (wikidata_dir / "items.txt").open("r") as f:
        # item_data = pickle.load(f)
        id2name: Dict[str, List[str]] = {}
        for lc, line in tqdm.tqdm(enumerate(f), ncols=80, desc="Loading entities", ascii=True, total=37899556):
            #if lc > 10000:
            #    break
            line = line[:-1].split("\t")
            if len(line) == 3:
                k, name, aka = line
            else:
                k, name, aka, _ = line
            # id2name[k] = [v["name"]] + v["aka"]
            id2name[k] = [name] + aka.split("||")

    # with (wikidata_dir / "item_with_dates.bin").open("rb") as f:
    #     item_with_dates = pickle.load(f)
    #     id2date = {}
    #     for k, v in tqdm.tqdm(item_with_dates.items(), ncols=80, desc="Loading entities", ascii=True):
    #         id2date[k] = [expand_date(v), 0]
    id2date = {}

    LOGGER.info(f"Loaded names and dates.")

    # Harvest neighbors.
    id2nb: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with (wikidata_dir / f"triples.txt").open("r") as f:
        for lc, line in tqdm.tqdm(enumerate(f), ncols=80, desc="Preparing triples", ascii=True, total=270306417):
            #if lc > 10000:
            #    break
            subj, rel, obj = line[:-1].split("\t")
            if obj.startswith("Q"):
                id2nb[subj].append((rel, obj))

    # Load properties to convert relation ids into names
    with (wikidata_dir / "properties.bin").open("rb") as f:
        props = pickle.load(f)
        prop2name = {k: v["name"] for k, v in props.items()}
        # Add the special relation "type"
        prop2name["@TITLE@"] = "TITLE"

    wikidata_kg = WikidataGraph(id2name, id2nb, id2date)

    nbs = []
    with (output_dir / f"hiro_subgraph.jsonl").open("w") as fout:
        for id_ in tqdm.tqdm(ids, ncols=80, desc="searching"):
            eg = wikidata_kg.expand_graph_me(Node(id_, [], 0, ""), distance=2)
            node_str = json.dumps([[node.relations, node.id, node.depth, node.head_id] for node in eg])
            fout.write('{}\n'.format(node_str))
            #nbs.append(eg)

    #import pdb;pdb.set_trace()


    # for mode in tqdm.tqdm(dataset, ncols=80, total=3, ascii=True):
    #     shards = len(dataset[mode]) // 1000
    #     for shard in tqdm.trange(shards+1, ncols=80, ascii=True):
    #         slice_ = slice(1000*shard, min(1000*(shard+1), len(dataset[mode])))
    #         data_chunk = dataset[mode][slice_]
    #         canonical_chunk = canonical_forms[mode][slice_]
    #         m, f = parse_articles(
    #             data_chunk,
    #             canonical_chunk,
    #             wikidata_kg,
    #             match_aka=(not args.ignore_aka),
    #             match_title=True,
    #             match_date=True,
    #             match_lower=False,
    #             match_stopwords=False,
    #         )

    #         # remove not_founds
    #         m = [(i, k) for (i, k, l) in m]

    #         # Missed articles
    #         with (output_dir / f"{mode}_{shard:02d}_NOT_FOUNDS.json").open("w") as fout:
    #             json.dump(f, fout, ensure_ascii=False, indent=2)

    #         # Insert show_example function here.
    #         named_matches = convert_ids(m, prop2name)

    #         output = []
    #         for article, relations in tqdm.tqdm(
    #                 list(zip(data_chunk, named_matches)), ncols=80, ascii=True
    #         ):
    #             # save the triple annotations with the tokens
    #             output.append((article["tokens"].strip().split(" "), relations[-1]))

    #         # Dump the dataset splits with relation annotations
    #         with (output_dir / f"{mode}_{shard:02d}.pkl").open("wb") as f:
    #             pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis.")
    parser.add_argument("--wikitext-dir", type=str, help="", required=True)
    parser.add_argument("--extracted-dir", type=str, help="", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ignore-aka", action="store_true")
    parser.add_argument(
        "--logging-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logfile = output_dir / f"{str(datetime.now()).replace(' ', '_')}.log"

    if not output_dir.exists():
        output_dir.mkdir()
    LOGGER.info(f"Saving into {output_dir}.")

    logging.basicConfig(
        datefmt="%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.getLevelName(args.logging_level),
        filename=logfile,
    )
    main(args)
