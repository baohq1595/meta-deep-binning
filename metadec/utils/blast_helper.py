import json
import re
import glob
from collections import defaultdict
import numpy as np
import os

from sklearn.utils.linear_assignment_ import linear_assignment
import itertools

def get_ref_subject(raw_name, raw_score):
    # Parse and remove unnecessary prefix from raw name
    pos_list = [pos.start() for pos in re.finditer(r'\|', raw_name)]
    name = raw_name[pos_list[-1] + 2:]
    pos_list = [pos.start() for pos in re.finditer(r'Length=', name)]
    name = name[:pos_list[-1]]

    trailing = re.search(r'[0-9]+ ?genomic scaffold.*', name)
    if trailing:
        name = name.replace(trailing.group(0), '')
    # pos_list = [pos.start() for pos in re.finditer(r'[0-9]+genomic scaffold.*')]

    score = raw_score[8: raw_score.find(' ', 9) + 1]
    identity_score = re.search(r'\([0-9]+%\)', raw_score).group(0)
    identity_score = re.sub(r'\(?\)?%?', '', identity_score)

    return name, score, identity_score

def parse_blast_result(result_file, export_file):
    blast_result = open(result_file, 'r')
    lines = blast_result.readlines()
    lines = [line.strip() for line in lines]
    lines = list(reversed(lines))
    results = []

    while(len(lines) > 0):
        try:
            line = lines.pop()
            if line == '':
                continue
            if line[:6] == 'Query=': # new query
                query_title = line.replace('Query=', '')
                
                # Check whether there are hits or not
                # Get new line
                line = lines.pop()
                while(line[:5] != '*****' and line[:9] != 'Sequences'):
                    # read new line
                    line = lines.pop()
                
                # No hits found, go to next query
                if line[:5] == '*****':
                    continue
                else: # at least 1 hit
                    while(True): # new query
                        line = lines.pop()
                        if line == '':
                            continue
                        
                        if line[0] == '>':
                            # next lines are name of scaffold
                            # read and concat these line into specie name
                            name = line
                            line = lines.pop()
                            while(line != ''):
                                name = name + line
                                line = lines.pop()

                            # next lines are score
                            line = lines.pop()
                            score = ''
                            while(line != ''):
                                score = score + line
                                line = lines.pop()
                        
                            # Parse the result
                            name, score = get_ref_subject(name, score)
                            results.append((name, score))
                            # Continue to search more query
                            break
        except EOFError as e:
            # end of file
            break
    
    blast_result.close()
    species_count = extract_species(results)
    with open(export_file, 'w') as f:
        json.dump(species_count, f)

    return species_count

def dump_reads_w_label(result_file, ref_reads_file, export_mapping_file, read_fna_file):
    blast_result = open(result_file, 'r')
    lines = blast_result.readlines()
    lines = [line.strip() for line in lines]
    lines = list(reversed(lines))

    specie2read_headers = defaultdict(list)

    while(len(lines) > 0):
        try:
            line = lines.pop()
            if line == '':
                continue
            if line[:6] == 'Query=': # new query
                query_title = line.replace('Query= ', '')
                
                # Check whether there are hits or not
                # Get new line
                line = lines.pop()
                while(line[:5] != '*****' and line[:9] != 'Sequences'):
                    # read new line
                    line = lines.pop()
                
                # No hits found, go to next query
                if line[:5] == '*****':
                    continue
                else: # at least 1 hit
                    while(True): # new query
                        line = lines.pop()
                        if line == '':
                            continue
                        
                        if line[0] == '>':
                            # next lines are name of scaffold
                            # read and concat these line into specie name
                            name = line
                            line = lines.pop()
                            while(line != ''):
                                name = name + line
                                line = lines.pop()

                            # next lines are score
                            line = lines.pop()
                            score = ''
                            while(line != ''):
                                score = score + line
                                line = lines.pop()
                        
                            # Parse the result
                            name, score, identity = get_ref_subject(name, score)
                            if int(identity) >= 95:
                                print(identity)
                            specie2read_headers[name.strip()].append(query_title)
                            # Continue to search more query
                            break
        except EOFError as e:
            # end of file
            break
    
    blast_result.close()
    with open(export_mapping_file, 'w') as f:
        json.dump(specie2read_headers, f)

    with open(ref_reads_file, 'r') as f:
        read_lines = f.readlines()

    reads = []
    is_read_part = False
    read_frags = []
    header = ''
    header2raw_reads = {}
    new_header2raw_reads = {}
    for k, line in enumerate(read_lines):
        if '>' in line or k == len(read_lines) - 1:
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
            
    

def extract_species(blast_result):
    species_count = defaultdict(int)
    for res in blast_result:
        specie = res[0]
        species_count[specie] += 1

    return species_count

def precision(res, indices):
    all_cluster_precs = []
    specices_precision = defaultdict(float)
    for k, cluster_res in enumerate(res):
        idx = indices[k]
        cluster_prec = cluster_res[idx] / np.sum(res[k, :])

        all_cluster_precs.append(cluster_prec)

    # print('precision: ', all_cluster_precs)
    return np.array(all_cluster_precs)

def recall(res, indices):
    all_cluster_recalls = []
    for k, cluster_res in enumerate(res):
        idx = indices[k]
        n = res[:,idx]
        cluster_recall = cluster_res[idx] / np.sum(res[:,idx])

        all_cluster_recalls.append(cluster_recall)

    # print('recall: ', all_cluster_recalls)
    return np.array(all_cluster_recalls)

def metrics(count_results):
    matrix = np.array(count_results)
    base_indices = [0, 1, 2, 3, 4]

    matrix = matrix + 1

    f1s = {}
    precisions = {}
    recalls = {}
    all_indices = list(itertools.permutations(base_indices))
    for i, indices in enumerate(all_indices):
        prec = np.mean(precision(matrix, indices))
        rec = np.mean(recall(matrix, indices))

        f1 = 2*((prec*rec)/(prec+rec))
        f1s[i] = f1
        precisions[i] = prec
        recalls[i] = rec

    max_f1 = 0
    max_id = None
    
    for f1_item in f1s.items():
        if f1_item[1] > max_f1:
            max_f1 = f1_item[1].item()
            max_id = f1_item[0]

    # return max_f1, np.mean(precisions[max_id]), np.mean(recalls[max_id]), all_indices[max_id], f1s[max_id]
    return max_f1, precisions[max_id], recalls[max_id], all_indices[max_id], f1s[max_id]

def calculate_metrics(result_dir):
    species = ['Thermoplasmatales archaeon Gpl Thermo_Gpl_Scaffold',
    'Ferroplasma acidarmanus Type I Ferro_acid_Scaffold',
    'Leptospirillum sp. Group III LeptoIII_Scaffold',
    'Ferroplasma sp. Type II FerroII_Scaffold',
    'Leptospirillum sp. Group II \'5-wayCG\' Scaffold']

    result_files = glob.glob(result_dir + '/*.json')
    result_counts = []
    for result_file in result_files:
        if 'final_result' in result_file:
            continue
        with open(result_file, 'r') as f:
            result_count = json.load(f)
            result_counts.append(result_count)
    
    combined_results = defaultdict(list)
    result_matrix = []
    # for each cluster
    for i, result_count in enumerate(result_counts):
        # for each specie
        count_list = []
        for j, specie in enumerate(species):
            combined_results[specie].append(result_count.get(specie, 0))
            count_list.append(result_count.get(specie, 0))
        result_matrix.append(count_list)

    max_f1, precision, recall, indices, f1s = metrics(result_matrix)
    np.set_printoptions(precision=2, suppress=True)
    
    log = {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': max_f1,
        'mapping indices': indices
    }

    with open(os.path.join(result_dir, 'final_result.json'), 'w') as f:
        json.dump(log, f)

    # print('f1: ', max_f1)
    # print('idx: ', indices)
    # print('f1s: ', f1s)


    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", help="Directory contains clustered results",
                        type=str)

    # args = parser.parse_args()
    # res_dir = args.result_dir
    # res_dir = 'C:/Users/ASUS/Desktop/fnas\lr0.0001_preoptimsgd_prelr0.01_drop0.0/blast_result'
    root_dir = 'E:/workspace/python/metagenomics-deep-binning/data/amd/Amdfull'
    amd_file = 'E:\\workspace\\python\\metagenomics-deep-binning\\data\\amd\\Amdfull\\raw_amd.fna'
    # all_dirs = glob.glob(root_dir + '/*')
    all_dirs = [root_dir]
    for res_dir in all_dirs:
        print('Processing folder ', res_dir)
        # res_dir = res_dir + '/blast_result'
        all_files = glob.glob(res_dir + '/*.txt')
        parsed_dir = os.path.join(res_dir, 'parsed')
        if not os.path.exists(parsed_dir):
            os.makedirs(parsed_dir)
        for file in all_files:
            print('Parsing ', file)
            results = dump_reads_w_label(file, amd_file,
                                        os.path.join(parsed_dir, os.path.basename(file).split('.')[0] + '.json'),
                                        os.path.join(parsed_dir, 'taxo_amd.fna'))

        # calculate_metrics(parsed_dir)
