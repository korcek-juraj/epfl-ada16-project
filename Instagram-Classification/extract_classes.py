import argparse
import json


def extract_classes(results_filename, classes_filename):
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


