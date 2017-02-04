import pickle
import multiprocessing
from multiprocessing import Process, Queue, Value
from queue import Empty
from urllib import request, error
import os.path
from ctypes import c_bool
import json
import time
import argparse

import tensorflow as tf
import numpy as np

from classify_image import maybe_download_and_extract, create_graph, NodeLookup, FLAGS


def download_worker(url_queue, image_queue):
    while True:
        try:
            img_id, url_list = url_queue.get(timeout=1)
            if img_id is None:
                break
            filename = 'images/' + img_id + '.jpg'
            if os.path.isfile(filename):
                image_queue.put(filename)
                continue
            else:
                for url in url_list:
                    try:
                        request.urlretrieve(url, filename)
                        break
                    except (error.HTTPError, error.URLError):
                        continue
                if not os.path.isfile(filename):
                    open(filename + '.failed','a+').close()
                    print(img_id + ': image missing!', flush=True)
                else:
                    image_queue.put(filename)
        except Empty:
            continue
        except Exception as e:
            print('Unexpected exception in download_worker: ' + str(e), flush=True)


def classification_worker(image_queue, result_queue):
    create_graph()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        while True:
            try:
                image = image_queue.get(timeout=1)
                if image is None:
                    break
                img_id, result = predict(image, sess, softmax_tensor)
                result_queue.put({img_id: result})
                os.remove(image)
            except Empty:
                continue
            except Exception as e:
                print('Unexpected exception in classification_worker: ' + str(e), flush=True)


def predict(image, sess, softmax_tensor):
    img_id = os.path.splitext(os.path.basename(image))[0]
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    result = {}
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        result[human_string] = float(score)

    return img_id, result


def save_result_worker(result_queue, output_file):
    with open(output_file, "a+") as result_file:
        while True:
            try:
                kv = result_queue.get(timeout=1)
                if kv is None:
                    break
                result_file.write(json.dumps(kv) + '\n')
            except Empty:
                continue


def progress_reporting_worker(url_queue, url_queue_orig_len, image_queue, result_queue, classification_done, progress_report_interval):
    while not bool(classification_done.value):
        print('Images remaining (length of url_queue): ' + str(url_queue.qsize()) + '/' + str(url_queue_orig_len), flush=True)
        print('image_queue length: ' + str(image_queue.qsize()), flush=True)
        print('result_queue length: ' + str(result_queue.qsize()), flush=True)
        time.sleep(progress_report_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_imgs_pickle', type=str)
    parser.add_argument('-o', '--output_json_file', type=str)
    parser.add_argument('-dp', '--download_process_count', type=int, default=2)
    parser.add_argument('-cp', '--classification_process_count', type=int, default=3)
    parser.add_argument('-t', '--test_run', type=int, default=0)
    parser.add_argument('-ri', '--progress_report_interval', type=int, default=5)
    args = parser.parse_args()
    download_process_no = args.download_process_count
    classification_process_no = args.classification_process_count
    test_run = args.test_run  # no of img_ids to process, None or 0 if all
    progress_report_interval = args.progress_report_interval  # seconds, if None or 0 => no progress reporting
    if args.output_json_file:
        output_file = args.output_json_file
    else:
        output_file = 'result' + time.strftime("%Y%m%d-%H%M%S") + '.json'

    ctx = multiprocessing.get_context('spawn')

    maybe_download_and_extract()

    with open(args.path_to_imgs_pickle, 'rb') as handle:
        urldict = pickle.load(handle)

    imgid_urls_queue = ctx.Queue()
    i = 0
    for img_id, url_list in urldict.items():
        i += 1
        imgid_urls_queue.put((img_id, url_list))
        if test_run and i >= test_run:
            break
    imgid_urls_queue_orig_len = imgid_urls_queue.qsize()

    img_filename_queue = ctx.Queue()

    download_p_list = []
    for i in range(1, download_process_no + 1):
        download_p = ctx.Process(target=download_worker, args=(imgid_urls_queue, img_filename_queue,))
        download_p.daemon = True
        download_p_list.append(download_p)
        imgid_urls_queue.put((None, None))
        download_p.start()

    result_queue = ctx.Queue()
    classification_done = ctx.Value(c_bool, False)

    classification_p_list = []
    for i in range(1, classification_process_no + 1):
        classification_p = ctx.Process(target=classification_worker, args=(img_filename_queue, result_queue,))
        classification_p.daemon = True
        classification_p_list.append(classification_p)
        classification_p.start()

    save_results_p = ctx.Process(target=save_result_worker, args=(result_queue, output_file,))
    save_results_p.daemon = True
    save_results_p.start()

    if progress_report_interval:
        progress_reporting_p = Process(target=progress_reporting_worker, args=(imgid_urls_queue, imgid_urls_queue_orig_len, img_filename_queue, result_queue, classification_done, progress_report_interval,))
        progress_reporting_p.daemon = True
        progress_reporting_p.start()

    for p in download_p_list:
        p.join()

    print('Download done!', flush=True)

    for p in classification_p_list:
        img_filename_queue.put(None)

    for p in classification_p_list:
        p.join()

    classification_done.value = True
    print('Classification done!', flush=True)

    result_queue.put(None)

    save_results_p.join()

    if progress_report_interval:
        progress_reporting_p.join()

    print('Done!')
