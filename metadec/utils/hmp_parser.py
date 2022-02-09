import json
import re
import glob
from collections import defaultdict
import numpy as np
import os

def get_ref_subject_sharon(raw_name, raw_score):
    name = raw_name[len('NZ_FR903368.1 '):]

    trailing = re.search(r' genomic scaffold.*', name)
    if trailing:
        name = name.replace(trailing.group(0), '')

    score = raw_score[8: raw_score.find(' ', 9) + 1]

    return name, score

def dump_reads_w_label(result_file, ref_reads_file, export_mapping_file, read_fna_file):
    blast_result = open(result_file, 'r')
    # lines = blast_result.readlines()
    # lines = [line.strip() for line in lines]
    # lines = list(reversed(lines))

    specie2read_headers = defaultdict(list)

    # while(len(lines) > 0):
    # not_break = True
    while(True):
    # for line in blast_result:
        try:
            line = blast_result.readline()
            # print(line)
            if line == '\0':
                break
            # line = lines.pop()
            if line == '':
                continue
            if line[:6] == 'Query=': # new query
                query_title = line.replace('Query= ', '')
                
                # Check whether there are hits or not
                # Get new line
                # line = lines.pop()
                line = blast_result.readline()
                while(line[:5] != '*****' and line[:9] != 'Sequences'):
                    # read new line
                    # line = lines.pop()
                    line = blast_result.readline()
                
                # No hits found, go to next query
                if line[:5] == '*****':
                    continue
                else: # at least 1 hit
                    while(True): # new query
                        # line = lines.pop()
                        name_score = []
                        line = blast_result.readline()
                        if line == '':
                            continue
                        elif 'Effective search space' in line:
                            break
                        elif '>' not in line:
                            continue
                        
                        if line[0] == '>':
                            # next lines are name of scaffold
                            # read and concat these line into specie name
                            name = line
                            # line = lines.pop()
                            line = blast_result.readline()
                            while(line != ''):
                                name = name + line
                                # line = lines.pop()
                                line = blast_result.readline()

                            # next lines are score
                            # line = lines.pop()
                            line = blast_result.readline()
                            score = ''
                            while(line != ''):
                                score = score + line
                                # line = lines.pop()
                                line = blast_result.readline()
                        
                            # Parse the result
                            name, score = get_ref_subject_sharon(name, score)
                            name_score.append((name, score))

                    # Assign this read to specie with highest score
                    name = sorted(name_score, key=lambda x: int(x[1]))[-1]
                    specie2read_headers[name.strip()].append(query_title)
                    # Continue to search more query
                    # break
        except EOFError as e:
            # end of file
            break
    
    blast_result.close()
    with open(export_mapping_file, 'w') as f:
        json.dump(specie2read_headers, f)

    # with open(ref_reads_file, 'r') as f:
    #     read_lines = f.readlines()
    f = open(ref_reads_file, 'r')

    reads = []
    is_read_part = False
    read_frags = []
    header = ''
    header2raw_reads = {}
    new_header2raw_reads = {}
    for line in f:
        if '>' in line or line == '' or line == '\0':
            if len(read_frags) > 0:
                header2raw_reads[header] = ''.join(read_frags)
                read_frags = []
            if '>' in line:
                header = line.strip().replace('>', '')
        else:
            read_frags.append(line)

    seq_count = 1
    count = 0
    for specie, blasted_headers in specie2read_headers.items():
        count += len(blasted_headers)
        print(len(blasted_headers))
        for header in blasted_headers:
            new_header = '>' + header.strip() + '|' + specie + '\n'
            new_header2raw_reads[new_header] = header2raw_reads[header]

    print(len(list(header2raw_reads.keys())))

    with open(read_fna_file, 'w', newline='') as f:
        for header, read in new_header2raw_reads.items():
            f.write(header)
            f.write(read)

    print('Blasted ratio: ', count / len(header2raw_reads))


if __name__ == "__main__":
    blast_result_file = 'E:/workspace/python/metagenomics-deep-binning/data/hmp/SRR1804065_1.txt'
    read_file = 'E:/workspace/python/metagenomics-deep-binning/data/hmp/SRR1804065_1.fasta'
    parsed_dir = 'E:/workspace/python/metagenomics-deep-binning/data/hmp'
    results = dump_reads_w_label(blast_result_file, read_file,
                                os.path.join(parsed_dir, os.path.basename(blast_result_file).split('.')[0] + '.json'),
                                os.path.join(parsed_dir, 'taxo_SRR1804065_1.fna'))