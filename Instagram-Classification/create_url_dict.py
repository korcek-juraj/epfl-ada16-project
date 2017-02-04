import os
import argparse
import pickle


def create_url_dict(filename, append_to_dict=None):
    """
    Processes the file passed in filename argument.
    The file is expected to contain key-value pairs (split by space) of Instagram image ID and its url at every line.

    The result is a dictionary with every Instagram Image ID as key and list of urls associated to it as value.

    The parsed results can be returned as a new dictionary or be appended to an existing one passed in argument 'append_to_dict'
    :param filename: a file with Instagram IDs and urls to process
    :param append_to_dict: a dictionary to append the results to; if None, new dictionary is returned
    :return: new dictionary or modified dictionary passed in 'append_to_dict' parameter
    """
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


def main(filename):
    """
    Parses file passed in filename using create_url_dict method and pickles resulting dictionary for further use.

    Resulting pickle file is saved into the same directory as the input file with the same filename,
    but with extension '.pickle'.
    :param filename: filename of file to be parsed
    :return: None
    """
    filename, ext = os.path.splitext(args.filename)
    urldict = create_url_dict(filename + ext)
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(urldict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True)
    args = parser.parse_args()

    main(args.filename)
