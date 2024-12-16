import csv
import os
import json
import re
import string
from pprint import pprint


def get_data_by_field(data, field):
    return [e[field] for e in data[0]['fields'] if e.get(field) is not None]


def extract_title(data):
    """提取标题"""
    values = get_data_by_field(data, '245')
    # pprint(values)
    assert len(values) == 1
    title = []
    c_end_char =  None
    has = False
    for e in values[0]['subfields']:
        if 'c' in e:
            c_end_char = e['c'][-1]
            if c_end_char not in string.punctuation:
                c_end_char = None
            if '=' in e['c'] and '/' in e['c']:
                idx1 = e['c'].index('=')
                idx2 = e['c'].index('/')
                if -1< idx1 < idx2:
                    if len(title) > 0 and title[-1].endswith('/'):
                        title[-1] = title[-1][0:-1].strip()
                    title.append("= "+(e['c'][idx1+1:idx2].strip()))
                    has = True
            continue
        sub_values = [se for se in e.values()]
        assert len(sub_values) == 1
        if c_end_char:
            title.append(c_end_char)
            c_end_char = None
            has = True
        title.append(sub_values[0].strip())
    title = ' '.join(title)
    title = title.strip()
    if title.endswith('/'):
        title = title[0:-1].strip()
    # if has:
        # pprint(data)
        # print(title)
    return title


def extract_abstract(data):
    """提取摘要"""
    values = get_data_by_field(data, '520')
    abstract = []
    for value in values:
        ind1 = value['ind1'].strip()
        if not (ind1 == '' or ind1 == '3'):
            continue
        for e in value['subfields']:
            sub_values = [se for se in e.values()]
            assert len(sub_values) == 1
            abstract.append(sub_values[0].strip())
    return '\n'.join(abstract)



def lcc_standarize(lcc):
    # idx, end_idx = -1, -1
    # for i in range(0, len(lcc)):
    #     if lcc[i].isalpha():
    #         idx = i
    #     else:
    #         break
    # if idx>-1:
    #     for i in range(idx, len(lcc)):
    #         if lcc[i].isdigit() or lcc[i]=='.':
    #             idx = i
    #         else:
    #             if lcc[i-1] == '.' and lcc[i].isalpha():
    #                 end_idx = i-1
    #                 break
    # if end_idx > -1:
    #     lcc_std = lcc[0:end_idx]
    # else:
    #     lcc_std = lcc
    lcc_std = re.match(r'^([A-Za-z]+)\W*(\d+(\.\d+)?)', lcc)
    if lcc_std:
        lcc_std = lcc_std.group()
        lcc_std = lcc_std.upper().replace(" ", "")
    else:
        lcc_std = ''
        # print(lcc)
    return lcc_std


def extract_lcc(data):
    """提取国会图书馆分类号"""
    values1 = get_data_by_field(data, '050')
    values2 = get_data_by_field(data, '090')
    values = values1+values2
    lccs = []
    lccs_std = []
    for value in values:
        for e in value['subfields']:
            if 'a' in e:
                sub_values = [se for se in e.values()]
                assert len(sub_values) == 1
                lcc = sub_values[0].strip()
                lccs.append(lcc)
                lccs_std.append(lcc_standarize(lcc))
    return ' ; '.join(lccs), ' ; '.join(lccs_std)


def extract_table_of_contents(data):
    """提取目录"""
    values = get_data_by_field(data, '505')
    tocs = []
    for value in values:
        for e in value['subfields']:
            sub_values = [se for se in e.values()]
            assert len(sub_values) == 1
            tocs.append(sub_values[0].strip())
    assert len(tocs) > 0
    return '\n'.join(tocs).strip()


def extract_publisher_year(data):
    """提取年份"""
    values = get_data_by_field(data, '008')
    if len(values) > 0:
        # if values[0][7:11] !=  values[0][11:15] and values[0][11:15].strip():
        #     print(values)
        return values[0][7:11].strip(), values[0][11:15].strip()
    else:
        values = get_data_by_field(data, '264')
        assert len(values) == 1
        if values[0]['ind2'] == '1':
            for e in values[0]['subfields']:
                if 'c' in e:
                    sub_values = [se for se in e.values()]
                    assert len(sub_values) == 1
                    year = re.search(r'\d+', sub_values[0]).group()
                    # print("------")
                    # print(sub_values[0], year)
                    return year, ''
        return None, None


def extract_subject_headings(data):
    """提取主题词"""
    fields = [
        '650', # Topical subject heading
        '600', # Personal name
        '610', # organizational name
        '611', # meeting name
        '630', # Title
        '655', # Genre heading
        '647', # Named event
        '648', # chronological term
        '651', # geographic name  
    ]
    subjects_lcsh = []
    subjects_fast = []
    for field in fields:
        values = get_data_by_field(data, field)
        for value in values:
            if value['ind2'].strip() == '0': # LCSH headings
                subject = [e['a'].strip() for e in value['subfields'] if e.get('a')]
                subjects_lcsh.extend(subject)
            elif value['ind2'].strip() == '7': # FAST headings
                has_fast = [e['2'].strip() for e in value['subfields'] if e.get('2')]
                if len(has_fast) >= 1 and has_fast[0].lower() == 'fast': # FAST headings
                    subject = [e['a'].strip() for e in value['subfields'] if e.get('a')]
                    subjects_fast.extend(subject)
    return '; '.join(subjects_lcsh), '; '.join(subjects_fast)


def extract_bibli(data_dir, save_file_path):
    with open(save_file_path, mode='w', newline='', encoding='utf-8-sig') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['lcc', 'lcc_std', 'start_year', 'end_year', 'title', 'abstract', 'toc', 'lcsh_subject_headings', 'fast_subject_headings'])
        filenames = os.listdir(data_dir)
        filenames.sort()
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(data_dir, filename), mode='r') as infile:
                data = infile.read()
                data = json.loads(data)
                title = extract_title(data)
                abstract = extract_abstract(data)
                lcc, lcc_std = extract_lcc(data)
                toc = extract_table_of_contents(data)
                syear, eyear = extract_publisher_year(data)
                lcsh_subject_headings, fast_subject_headings = extract_subject_headings(data)
                if lcc_std:
                    writer.writerow([lcc, lcc_std, syear, eyear, title, abstract, toc, lcsh_subject_headings, fast_subject_headings])
                else:
                    print(lcc, lcc_std)
                # print(lcc, lcc_std)
                # if ' ' in lcc:
                # if len(lcc.split("."))>3:
                    # print(lcc, lcc_std)
                # print(lcc,'\t\t', lcc_std, syear, eyear)
                # if ';' in lcc:
                #     print(lcc)
                # pprint(data)
                # print(syear, eyear)
                # break


if __name__=='__main__':
    data_dir = '/home/hong/bookcls/data/json'
    save_file_path = '/home/hong/bookcls/bibli.csv'
    extract_bibli(data_dir, save_file_path)
    print("ok")
    # print(lcc_standarize('Z711.6.G46'))