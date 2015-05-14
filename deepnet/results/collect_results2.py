from deepnet import util
from deepnet import deepnet_pb2
import os, sys
import glob
from collections import defaultdict
from collections import namedtuple
from itertools import product
from deepnet import markup
from operator import itemgetter
import webbrowser

output_directories = ['dbm_BEST']
output_file_suffixes = ['BEST', 'LAST']

# Drop any files that contain this stri ng
drop_string_list = ['rbm2', 'rbm1_imperr_BEST']


def walk_dir(dirpath):
    model_paths = defaultdict(list)
    for root, dirs, files in os.walk(dirpath):
        for f,suffix in product(files, output_file_suffixes) :
            if f.endswith(suffix): 
                model_paths[suffix].append(os.path.join(root, f))
    return dict(model_paths) 

def make_table(page, rows, header=None):
    """ Make an HTML table for a markup page """
    page.table(border=1)
    page.tr()
    page.th(header)
    page.tr.close()
    for row in rows:
        page.tr()
        page.td(row)
        page.tr.close()
    page.table.close()

def encode_html_rows(rows, header):
    """ Encode rows into html """
    page = markup.page()
    page.h2("Job Table")
    page.init("Job Table")
    make_table(page, rows, header= header)
    return page.__str__()

def display_html(rows, header, launch_browser=False):
    """ Display all table results """
    html = encode_html_rows(rows, header)

    outf = os.path.join("/tmp", os.urandom(16).encode('hex')+".html")
    with open(outf, 'w') as fout:
        fout.write(html)
    
    if launch_browser:
        webbrowser.open(outf, new=2)

DisplayRecord = namedtuple('DisplayRecord', 'Train_CE, Valid_CE, Test_CE, '+\
        'Train_Acc, Valid_Acc, Test_Acc, model_name, dataset, '+\
        'model_type, '+\
        'best_step, sparsity, dropout, '+ \
        'input_width, bernoulli_hidden1_width, '+\
        'bernoulli2_hidden1_width, gaussian_hidden1_width, '+\
        'epsilon, epsilon_decay, l2_decay, '+\
        'initial_momentum, final_momentum, expid')


def create_display_row(expid, model, op, model_type):
    hyperparams =  model.hyperparams

    record = []

    train_stat = model.train_stats[-1]
    valid_stat = model.validation_stats[-1]
    test_stat = model.test_stats[-1]

    for stat in (train_stat, valid_stat, test_stat):
        record.append("{:.5f}".format(stat.cross_entropy / stat.count))

    for stat in (train_stat, valid_stat, test_stat):
        record.append(stat.correct_preds / stat.count)

    record.append(model.name)
    record.append(op.data_proto_prefix.split("/")[-1])
    record.append(model_type)
    record.append(op.current_step)
    record.append(hyperparams.sparsity)
    record.append(hyperparams.dropout)
    
    for layer_name in ('input_layer', 'bernoulli_hidden1', 'bernoulli2_hidden1', 'gaussian_hidden1'):
        try:
            layer = next(l for l in model.layer if l.name == layer_name)
            record.append(layer.dimensions)
        except StopIteration:
            record.append("None")
    record.append("{:.4f}".format(hyperparams.base_epsilon))

    if hyperparams.epsilon_decay == deepnet_pb2.Hyperparams.INVERSE_T:
        record.append("INVERSE_T")
    elif hyperparams.epsilon_decay == deepnet_pb2.Hyperparams.EXPONENTIAL:
        record.append("EXPONENTIAL")
    elif hyperparams.epsilon_decay == deepnet_pb2.Hyperparams.NONE:
        record.append("NONE")
    else:
        raise ValueError("Unknown decay type")

    record.append("{:.4f}".format(hyperparams.l2_decay))
    record.append("{:.4f}".format(hyperparams.initial_momentum))
    record.append("{:.4f}".format(hyperparams.final_momentum))
    record.append(expid)
    
    return record
#    return DisplayRecord(*map(str, record))

def get_field_idx(name):
    return next(idx for (idx, field) in enumerate(DisplayRecord._fields) if field == name)

def sort_by_fields(records, fieldnames = []):
    idxs = map(get_field_idx, fieldnames)
    
    selected_fields = [tuple([r[idx] for idx in idxs]) for r in records]
    sorted_idxs = [idx for (idx,_) in sorted(enumerate(selected_fields), key= itemgetter(1))]

    return [records[idx] for idx in sorted_idxs]

def write_csv(records, headers, outf):
    import csv 
    with open(outf, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for record in records:
            writer.writerow(record)


if __name__ == '__main__':  
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Parses results by walking directories')
    parser.add_argument("--outf", type=str, help="Output file")
    parser.add_argument("--mode", type=str, help="html/csv")
    args = parser.parse_args()
    model_paths = walk_dir('.')
    exp_paths = defaultdict(list)

    get_expid = lambda f : f.split("/")[1]
    get_model = lambda f : util.ReadModel(f)
    get_op = lambda f : util.ReadOperation(f)

    no_match = lambda path : all(bool(s not in path) for s in drop_string_list)
    model_paths['BEST'] = filter(no_match, model_paths['BEST'])
        

    for path in model_paths['BEST']:
        exp_paths[get_expid(path)].append(path)
    exp_paths = dict(exp_paths)

    rows = []
    for exp in exp_paths:
        models = defaultdict(list)
        for f in exp_paths[exp] :
           model_name = os.path.basename(f).split('_')[0]
           models[model_name].append(f)
        models = dict(models)

        for name, (model_file, op_file) in models.items():

            # Get model type
            readme_path = "/".join(model_file.split("/")[:-2])
            readme_path = os.path.join(readme_path, 'README')

            with open(readme_path) as fin:
                readme_args = fin.read()
            tup = readme_args.split()
            idx = next(idx for idx,e in enumerate(tup) if '--model' == e)
            model_type = tup[idx+1]

            if 'train' in model_file : model_file,op_file = op_file, model_file
            row = create_display_row(exp, util.ReadModel(model_file), util.ReadOperation(op_file),\
                    model_type)
            rows.append(row)

        rows = sort_by_fields(rows, fieldnames=['dataset', 'Valid_CE'])
    if args.mode == 'html' :
        display_html(rows, DisplayRecord._fields, launch_browser=True) 
    elif args.mode == 'csv':
        if not args.outf:
            raise ValueError('CSV file destination needed')
        write_csv(rows, DisplayRecord._fields, args.outf)
    else:
        raise ValueError("Unknown output mode")

