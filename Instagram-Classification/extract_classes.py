import argparse
import json


def extract_classes(results_filename, classes_filename):
    """
    Reads results JSON file line by line and extracts classes / objects found in every image, i.e., it does groupby by
    the class name while aggregating the count for the class and summing over the confidence of the particular class prediction.

    The output file is in form of JSON list of tuples (Image ID, {'count': count, 'score': score, 'weight':0}) sorted by
    highest score (sum of confidence). Weight stands for sentiment of the particular class. It is set to 0 and is supposed
    to be manually changed to values between -1 and 1, before the file can be used for sentiment calculation using
    calculate_sentiment.py script.
    :param results_filename: filename of the results JSON file to extract classes from
    :param classes_filename: filename of a JSON file where the extracted classes will be written
    :return: None
    """
    classes_dict = {}
    img_id_cache = set()
    with open(results_filename) as results_file:
        for line in results_file:
            result_dict = json.loads(line)
            img_id = next(iter(result_dict))
            if img_id not in img_id_cache:
                img_id_cache.add(img_id)
                for cls, score in result_dict[img_id].items():
                    if cls not in classes_dict:
                        classes_dict[cls] = {'weight': 0, 'score': 0, 'count': 0}
                    classes_dict[cls]['score'] += score
                    classes_dict[cls]['count'] += 1
    with open(classes_filename, 'wt') as classes_file:
        json.dump(sorted(list(classes_dict.items()), key=lambda x: x[1]['score'], reverse=True), classes_file, sort_keys=True, indent=4, separators=(',', ': '))
    print('No. of classses: ' + str(len(classes_dict)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path_to_results_file', type=str)
    parser.add_argument('-c', '--output_classes_file', type=str)
    args = parser.parse_args()
    extract_classes(args.path_to_results_file, args.output_classes_file)


