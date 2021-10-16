import os
import csv
import json
import random

def convert_ori_to_pair_a_line():
    root_path = "/home/jiangjinhao/work/QA/ComQA/Commonsense-Path-Generator-main/commonsense-qa/data/snli"
    input = ["dev.tsv", "test.tsv", "train.tsv"]
    output = ["dev.txt", "test.txt", "train.txt"]
    for i,o in zip(input, output):
        op = os.path.join(root_path, o)
        ip = os.path.join(root_path, i)
        with open(op, "w") as opf:
            with open(ip,"r") as ipf:
                data = csv.reader(ipf, delimiter="\t")
                for idx, row in enumerate(data):
                    if idx == 0:
                        continue
                    uniq_id = row[2]
                    if len(row) == 15:
                        s1, s2 = row[7], row[8]
                        label = row[14]
                    else:
                        s1, s2 = row[7], row[8]
                        label = row[len(row)-1]
                    line = uniq_id + "\t" + s1 + "\t" + s2 + "\t" + str(label) + "\n"
                    opf.write(line)
    print("Complete %s , and output save in %s"%(ip,op))

def construct_nli_for_inter_pretrain():
    root_path = "/home/jiangjinhao/work/QA/ComQA/Commonsense-Path-Generator-main/commonsense-qa/data/snli/"
    for t in ["train","dev","test"]:
        path = root_path + t + ".txt"
        out_path = root_path + t + "_postprocess_choose_1.txt"
        with open(path, "r") as f:
            examples = []
            last_prefix = None
            sentences_count = 0
            samples_count = 0
            triples_count = 0
            all_input_count = 0
            line = f.readline().strip("\n")
            sentences_count += 1
            while line:
                id, pre, hyp, label = [s.strip() for s in line.split("\t")]
                prefix = pre
                if prefix != last_prefix:
                    sample = {"p": [], "contradiction": [], "entailment": [], "neutral": []}
                    relation = set()
                    last_prefix = prefix
                    sample["p"].append(pre)
                    while prefix == last_prefix:
                        sample[label].append(hyp)
                        relation.add(label)
                        line = f.readline().strip("\n")
                        if not line:
                            break
                        sentences_count += 1
                        _, pre, hyp, label = [s.strip() for s in line.split("\t")]
                        prefix = pre
                    samples_count += 1
                # process one sample to construct (pre,hyp,entail) and (pre, hyp, neural)
                if len(relation) == 3:
                    triples_count += 1
                    precise = sample["p"][0]
                    # for c in sample["contradiction"]:
                    #     for e in sample["entailment"]:
                    #         for n in sample["neutral"]:
                    #             one_sample = {}
                    #             one_sample["p"] = precise
                    #             one_sample["c"] = c
                    #             one_sample["n"] = n
                    #             one_sample["e"] = e
                    #             examples.append(one_sample)
                    e = random.choice(sample["entailment"])
                    n = random.choice(sample["neutral"])
                    c = random.choice(sample["contradiction"])
                    one_sample = {}
                    one_sample["p"] = precise
                    one_sample["c"] = c
                    one_sample["n"] = n
                    one_sample["e"] = e
                    examples.append(one_sample)
                    all_input_count += 1
            print("Total sentence pair are: %d, samples are: %d, triplets are:%d, inputs are %d" % (
                sentences_count, samples_count, triples_count, all_input_count))
        with open(out_path, "w") as f:
            for e in examples:
                es = json.dumps(e)
                f.write(es)
                f.write("\n")
        print("Complete %s dataset"%(t))

def main():
    construct_nli_for_inter_pretrain()

if __name__ == '__main__':
    main()