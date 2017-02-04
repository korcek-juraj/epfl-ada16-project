import os
import argparse
import pickle


def create_url_dict(filename, append_to_dict=None):
    if append_to_dict is None:
        ret = {}
    else:
        ret = append_to_dict

    with open(filename) as urlfile:
        for line in urlfile:
            elem_list = line.split()
            if elem_list[0] in ret:
                ret[elem_list[0]].append(elem_list[1])
            else:
                ret[elem_list[0]] = [elem_list[1], ]
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str)
    args = parser.parse_args()

    filename = os.path.splitext(args.filename)[0]
    urldict = create_url_dict(filename + '.txt')
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(urldict, handle, protocol=pickle.HIGHEST_PROTOCOL)
