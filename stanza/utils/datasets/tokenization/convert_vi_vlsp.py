
import os

def write_file(output_filename, sentences, shard):
    with open(output_filename, "w") as fout:
        for sent_idx, sentence in enumerate(sentences):
            fout.write("# sent_id = %s.%d\n" % (shard, sent_idx))
            fout.write("# text = ")
            fout.write(" ".join(sentence))
            fout.write("\n")
            for word_idx, word in enumerate(sentence):
                fake_dep = "root" if word_idx == 0 else "dep"
                fout.write("%d\t%s\t%s" % ((word_idx+1), word, word))
                fout.write("\t_\t_\t_")
                fout.write("\t%d\t%s" % (word_idx, fake_dep))
                fout.write("\t_\t_\n")
            fout.write("\n")

def convert_file(input_filename, output_filename, shard, split_filename=None, split_shard=None):
    with open(input_filename) as fin:
        lines = fin.readlines()

    sentences = []
    for line in lines:
        words = line.split()
        words = [w.replace("_", " ") for w in words]
        sentences.append(words)

    if split_filename is not None:
        split_point = int(len(sentences) * 0.85)
        write_file(output_filename, sentences[:split_point], shard)
        write_file(split_filename, sentences[split_point:], split_shard)        
    else:
        write_file(output_filename, sentences, shard)

def convert_vi_vlsp(extern_dir, tokenizer_dir, args):
    input_path = os.path.join(extern_dir, "vietnamese", "VLSP2013-WS-data")

    input_train_filename = os.path.join(input_path, "VLSP2013_WS_train_gold.txt")
    input_test_filename = os.path.join(input_path, "VLSP2013_WS_test_gold.txt")
    if not os.path.exists(input_train_filename):
        raise FileNotFoundError("Cannot find train set for VLSP at %s" % input_train_filename)
    if not os.path.exists(input_test_filename):
        raise FileNotFoundError("Cannot find test set for VLSP at %s" % input_test_filename)

    output_train_filename = os.path.join(tokenizer_dir, "vi_vlsp.train.gold.conllu")
    output_dev_filename = os.path.join(tokenizer_dir,   "vi_vlsp.dev.gold.conllu")
    output_test_filename = os.path.join(tokenizer_dir,  "vi_vlsp.test.gold.conllu")

    convert_file(input_test_filename, output_train_filename, "train", output_dev_filename, "dev")
    convert_file(input_test_filename, output_test_filename, "test")

