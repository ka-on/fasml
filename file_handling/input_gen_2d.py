from file_handling import input_gen


def read_input(path):    # read fasta like format for sec structure
    sec_features = {}
    with open(path, 'r') as infile:
        line = infile.readline().rstrip('\n')
        while line:
            prot_id = line.lstrip('>')
            ss3 = infile.readline().rstrip('\n')
            ss8 = infile.readline().rstrip('\n')
            diso = infile.readline().rstrip('\n')
            if prot_id and ss3 and ss8 and diso:
                sec_features[prot_id] = (ss3, ss8, diso)
            else:
                raise Exception('Error in structure file:\n' + path)
            line = infile.readline()
    return sec_features


def to_fas_format(sec_fasta_format):    # takes fasta like format and turns it into fas feature format
    sec_fas_format = {}
    for prot_id in sec_fasta_format:
        sec_fas_format[prot_id] = {'ss3': {'C_ss3': [], 'H_ss3': [], 'E_ss3': []},
                                   'ss8': {'C_ss8': [], 'H_ss8': [], 'E_ss8': [], 'S_ss8': [], 'T_ss8': [],
                                           'I_ss8': [], 'B_ss8': [], 'G_ss8': []},
                                   'disorder': {'D_disorder': []}}
        for stype in sec_fas_format[prot_id]:
            sec_fas_format = to_fas_single(sec_fasta_format[prot_id], sec_fas_format, stype, prot_id)
    sec_fas_format = {'feature': sec_fas_format}
    return sec_fas_format


def to_fas_single(sec_fasta_format, sec_fas_format, stype, prot_id):
    sdict = {'ss3': 0, 'ss8': 1, 'disorder': 2}
    current = None
    start = None
    for position in range(len(sec_fasta_format[sdict[stype]])):
        if not current:
            current = sec_fas_format[0][position]
            start = position + 1
        elif not sec_fas_format[0][position] == current:
            sec_fas_format[prot_id][stype][current + '_' + stype].append((start, position, None))
    sec_fas_format[prot_id][stype][current + '_' + stype].append((start, len(sec_fas_format[sdict[stype]]), None))
    return sec_fas_format


def regions_binary_2d(data, ss3):
    pids = [protein for protein in data["feature"]]
    if ss3 == 'ss3':
        features = ['C_ss3', 'H_ss3', 'E_ss3', 'D_disorder']
    else:
        features = ['C_ss8', 'H_ss8', 'E_ss8', 'S_ss8', 'T_ss8', 'I_ss8', 'B_ss8', 'G_ss8', 'D_disorder']
    regions = input_gen.shortest(data)
    matrix = input_gen.matrix_gen(data, pids, features, regions).T
    regions_dict = {}
    for row in matrix:
        regions_dict[row] = matrix[row].tolist()
    metadata = {
        "regions": regions,
        "features": features
    }
    output_data = {
        "data": regions_dict,
        "metadata": metadata
    }
    return output_data


def main(inpath, ss3):    # inpath: path to .struc file; ss3: boolean if true that use ss3 structure, else ss8
    sec_features = read_input(inpath)
    sec_fas_format = to_fas_format(sec_features)
    output_data = regions_binary_2d(sec_fas_format, ss3)
    return output_data
