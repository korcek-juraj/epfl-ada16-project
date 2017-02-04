import argparse
import json

import numpy as np


def calculate_sentiment(results_filename, classes_filename, sentiment_filename):
    sentiment_dict = {}

    with open(classes_filename) as classes_file:
        classes_list = json.load(classes_file)
        classes_dict = {cls: attr_dict['weight'] for cls, attr_dict in classes_list}

    with open(results_filename) as results_file:
        for line in results_file:
            result_dict = json.loads(line)
            img_id = next(iter(result_dict))
            if img_id not in sentiment_dict:
                sentiment_dict[img_id] = {'sent_int': 0, 'sent_float': 0}
                for cls, score in result_dict[img_id].items():
                    sentiment_dict[img_id]['sent_float'] += score * classes_dict[cls]
                sentiment_dict[img_id]['sent_int'] = (1 if abs(sentiment_dict[img_id]['sent_float']) > 0.33 else 0) * np.sign(sentiment_dict[img_id]['sent_float'])

    with open(sentiment_filename, 'wt') as sentiment_file:
        json.dump(sentiment_dict, sentiment_file, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path_to_results_file', type=str)
    parser.add_argument('-c', '--path_to_classes_file', type=str)
    parser.add_argument('-o', '--output_sentiment_file', type=str)
    args = parser.parse_args()
    calculate_sentiment(args.path_to_results_file, args.path_to_classes_file, args.output_sentiment_file)
