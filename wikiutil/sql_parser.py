import argparse, gzip
import logging
from tqdm import tqdm
from util import load_shyamupa_t2id

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)


def parse_schema(sqlfile, encoding):
    '''
    Parse the schema of the `sqlfile`
    '''
    schema = {}
    logging.info('Parsing schema for {}'.format(sqlfile))
    f = gzip.open(sqlfile, 'rt', encoding=encoding)
    start_parse = False
    fields = []
    for line in f:
        if start_parse:
            if 'PRIMARY KEY' not in line:
                fields.append(line)
            else:
                break
        else:
            if not line.startswith('CREATE TABLE'):
                continue
            else:
                start_parse = True
    f.close()
    for i, s in enumerate(fields):
        schema[s.split()[0][1:-1]] = i
    return schema


def split_str(split_char, split_str):
    '''
    This takes care of corner cases such as quotation: "xx,xx" and escape character,
    which will not be handled correctly by python split function.
    '''
    return_list = []
    i = 0
    last_pos = 0
    while i < len(split_str):
        if split_str[i] == '\'':
            i += 1
            while split_str[i] != '\'' and i < len(split_str):
                if split_str[i] == '\\':
                    i += 2
                else:
                    i += 1
        else:
            if split_str[i] == split_char:
                return_list.append(split_str[last_pos:i])
                last_pos = i + 1
        i += 1
    return_list.append(split_str[last_pos:])
    return return_list


def read_id2title(sqlfile, encoding, outpath, ln=0, constrain='0'):
    '''
    page id to wikipedia title
    '''
    logging.info('Reading id2title sql' + sqlfile)
    schema = parse_schema(sqlfile, encoding)
    logging.info('Schema {}'.format(schema))
    bad = 0
    if not ln:
        with gzip.open(sqlfile, 'rt', encoding=encoding) as f:
            for _ in f:
                ln += 1
    logging.info('Totally {} lines'.format(ln))
    with gzip.open(sqlfile, 'rt', encoding=encoding) as f, \
            open(outpath, 'w') as out:
        for line in tqdm(f, total=ln):
            if 'INSERT INTO' not in line:
                continue
            start = line.index('(')
            line = line[start + 1:]
            parts = line.split('),(')
            for part in parts:
                all_fields = split_str(',', part)
                if len(all_fields) != len(schema):
                    logging.warning('ignore part as number of fields does not match schema')
                    bad += 1
                    continue
                page_id = all_fields[schema['page_id']]
                ns = all_fields[schema['page_namespace']]
                page_title = all_fields[schema['page_title']]
                if ns != constrain:
                    continue
                is_redirect = all_fields[schema['page_is_redirect']]
                # strip out the single quotes around 'Title'
                page_title = page_title[1:-1]
                if '\\' in page_title:
                    page_title = page_title.replace('\\', '')
                buf = '\t'.join([page_id, page_title, is_redirect])
                out.write(buf + '\n')
        logging.warning('total bad formats in file: {}'.format(bad))


def read_pid2cate(sqlfile, encoding, outpath, ln=0, constrain='page'):
    '''
    page id to category
    '''
    logging.info('Reading pid2cate sql' + sqlfile)
    schema = parse_schema(sqlfile, encoding)
    logging.info('Schema {}'.format(schema))
    bad = 0
    if not ln:
        with gzip.open(sqlfile, 'rt', encoding=encoding) as f:
            for _ in f:
                ln += 1
    logging.info('Totally {} lines'.format(ln))
    with gzip.open(sqlfile, 'rt', encoding=encoding) as f, \
            open(outpath, 'wb') as out:
        for line in tqdm(f, total=ln):
            if 'INSERT INTO' not in line:
                continue
            start = line.index('(')
            line = line[start + 1:]
            parts = line.split('),(')
            for part in parts:
                all_fields = split_str(',', part)
                if len(all_fields) != len(schema):
                    logging.warning('ignore part as number of fields does not match schema')
                    bad += 1
                    continue
                pid = all_fields[schema['cl_from']]
                cate = all_fields[schema['cl_to']]
                cl_type = all_fields[schema['cl_type']]
                # only use page
                if cl_type[1:-1] != constrain:
                    continue
                # strip out the single quotes around cate
                cate = cate[1:-1]
                if '\\' in cate:
                    cate = cate.replace('\\', '')
                buf = '\t'.join([pid, cate])
                out.write((buf + '\n').encode(encoding))
        logging.warning('total bad formats {}'.format(bad))


def compress_pid2cate_mp(mapfile, cate2cateid, outpath):
    last_pid = '*'
    last_cate = []
    ne = 0
    all = 0
    with open(mapfile, 'r') as fin, open(outpath, 'w') as fout:
        for l in fin:
            l = l.strip()
            if len(l) == 0:
                continue
            all += 1
            pid, cate = l.split('\t')
            if cate not in cate2cateid:
                # skip nonexistent category
                ne += 1
                continue
            cid = cate2cateid[cate]
            if pid != last_pid and len(last_cate) > 0:
                fout.write('{}\t{}\n'.format(last_pid, ' '.join(map(str, last_cate))))
                last_cate = []
            last_pid = pid
            last_cate.append(cid)
        if len(last_cate) > 0:
            fout.write('{}\t{}\n'.format(last_pid, ' '.join(map(str, last_cate))))
    print('{} out of {} are nonexistent in cate2cateid map'.format(ne, all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a page id to cate map.')
    parser.add_argument('-task', type=str,
                        choices=['pid2cate', 'id2title', 'compress'], required=True)
    parser.add_argument('-data', type=str, required=True,
                        help='input file path')
    parser.add_argument('-out', type=str, required=True,
                        help='tsv file to write the map in.')
    args = parser.parse_args()
    if args.task == 'pid2cate':
        # The real encoding should be utf-8, use ISO-8859-1 to avoid bug
        # enwiki-20181101-categorylinks.sql.gz has 18026 lines
        # use 'page' or 'subcat' as constraints
        read_pid2cate(sqlfile=args.data, outpath=args.out,
                      encoding='ISO-8859-1', ln=0, constrain='page')
    elif args.task == 'id2title':
        # enwiki-20181101-page.sql.gz has 5457 lines
        read_id2title(sqlfile=args.data, outpath=args.out,
                      encoding='utf-8', ln=0, constrain='14')
    elif args.task == 'compress':
        cate2cateid = load_shyamupa_t2id('result/enwiki-20181020.cate2t')
        compress_pid2cate_mp(args.data, cate2cateid, args.out)
