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
import errno

import tensorflow as tf
import numpy as np
from classify_image import maybe_download_and_extract, create_graph, NodeLookup, FLAGS


def download_worker(url_queue, image_queue):
    """
    A worker for download processes.

    It executes following steps repeatedly:
    1) It gets Instagram Image ID and corresponding list of urls from "process-safe" queue passed in url_queue parameter.

    2) It attempts to download the image from urls available in the list into images/ folder as InstagramID.jpg file. If
    the image is successfully downloaded using an url, the rest of the urls for given Instagram Image ID in the list is
    ignored. If none of the urls worked, an empty file with name InstagramID.jpg.failed is created in the images/ directory
    for logging purposes.

    3) Once the image is downloaded its filename is put into the "process-safe" queue passed in image_queue parameter.
     From this queue classification workers obtain read-to-be-classified images.

    If an exception is thrown, it is logged and the worker continues to process next Instagram Image ID.

    The process / worker exits once it gets (None, None) tuple from the url_queue.
    :param url_queue: "process-safe" queue containing tuples (Instagram Image ID, list of image urls corresponding to given ID)
    :param image_queue: "process-safe" queue containing image filenames to be processed by classification workers
    :return: None
    """
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
    """
    A worker for classification processes.

    First, it loads TensorFlow Inception v3 CNN model using create_graph() function and initializes TensorFlow session.

    Then, it executes following steps repeatedly:
    1) It gets image filename from "process-safe" queue passed in image_queue parameter.

    2) It runs predictions on the image using predict() function.

    3) It deletes the image.

    4) The result is pushed onto "process-safe" queue passed in result_queue parameter in form of dictionary {img_id: result}.
     This queue feeds the process / worker responsible for saving the results into a file.

    If an exception is thrown, it is logged and the worker continues to process next image file.

    The process / worker exits once it gets None from the image_queue.
    :param image_queue: "process-safe" queue containing filenames of images to be classified
    :param result_queue: "process-safe" queue containing classification results to be processed by results-saving worker
    :return: None
    """
    create_graph()

    with tf.Session() as sess:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
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
    """
    Function used by classification workers to get prediction on image.

    This method was adapted based on run_inference_on_image() method from classify_image.py found in TensorFlow official tutorial.

    :param image: filename of the image to be classified
    :param sess: TensorFlow session
    :param softmax_tensor: tensor used for computing the predictions
    :return: (img_id, result) with img_id being Instagram Image ID and the result being dictionary with 5 most probable
    objects depicted in the image as keys and corresponding prediction confidences as values.
    """
    img_id = os.path.splitext(os.path.basename(image))[0]
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
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
    """
    A worker for process saving the classification results into a file.

    First, it opens/creates the outputfile specified in output_file parameter.

    Then, it executes following steps repeatedly:
    1) It gets image result dictionary from "process-safe" queue passed in result_queue parameter.

    2) It converts the result into JSON format and appends it to the output file.

    The process / worker exits once it gets None from the result_queue.
    :param result_queue: "process-safe" queue containing results from image classification in form of one dictionary per image
    :param output_file: filename of the results file to be written to
    :return: None
    """
    dir_path = os.path.dirname(output_file)
    try:
        os.makedirs(dir_path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass
        else:
            raise
    with open(output_file, "a+") as result_file:
        while True:
            try:
                kv = result_queue.get(timeout=1)
                if kv is None:
                    break
                result_file.write(json.dumps(kv) + '\n')
            except Empty:
                continue


def progress_reporting_worker(url_queue, url_queue_orig_len, image_queue, result_queue, progress_report_interval):
    """
    A simple worker for progress reporting and debugging purposes.

    It writes into the stdout number of images remaining to be processed.

    It also writes size of the two queues. This is useful to decide whether there is balance between number of download
    and classification workers.

    The process / worker is expected to be run in daemon mode and thus to exit once its parent process exits. Thus the
    infinite loops is not a problem.
    :param url_queue: "process-safe" queue containing tuples (Instagram Image ID, list of image urls corresponding to given ID)
    :param url_queue_orig_len: original length of url_queue
    :param image_queue: "process-safe" queue containing filenames of images to be classified
    :param result_queue: "process-safe" queue containing results from image classification in form of one dictionary per image
    :param progress_report_interval: interval how often to write the current progress into stdout in seconds
    :return: None
    """
    while True:
        print('Images remaining (length of url_queue): ' + str(url_queue.qsize()) + '/' + str(url_queue_orig_len), flush=True)
        print('image_queue length: ' + str(image_queue.qsize()), flush=True)
        print('result_queue length: ' + str(result_queue.qsize()), flush=True)
        time.sleep(progress_report_interval)


def main(path_to_imgs_pickle, test_run, download_process_no, classification_process_no, output_file, progress_report_interval):
    """
    Sets up and runs the multiprocessing pipeline.

    It consists of following steps:
    1) Downloading Inception v3 model if it is not present. This is done by maybe_download_and_extract() function.

    2) Filling imgid_urls_queue "process-safe" queue with Instagram Image IDs and its corresponding url lists from pickle
     file passed in path_to_imgs_pickle parameter. The amount of key value pairs (Image ID, url list) loaded is limited
     by the parameter test_run. Setting it to 0 or None loads all the available records form the file.

    3) Initialization of  "process-safe" queues img_filename_queue and result_queue.

    4) Spawning number of download process which is specified in download_process_no parameter.
    It also download_process_no-times appends tuples (None, None) to the end of imgid_urls_queue to signal the end of
    the queue for the download workers.

    5) Spawning number of classification process which is specified in classification_process_no parameter.

    6) Spawning worker for saving results into a file.

    7) Spawning progress-reporting worker.

    8) Joining the download workers. Once the download is done None is appened to the end of img_filename_queue
    classification_process_no-times to signal classification workers the end of queue.

    9) Joining classification workers. Once that is done, None is appended result_queue to signal results-saving worker
    the end of queue.

    10) Joining the results-saving worker.

    The progress reporting worker quits automatically with the main (parent) process because it is of daemon type.
    Therefore there is no need to join it.
    :param path_to_imgs_pickle: path to pickle file containing dictionary of Instagram Image IDs as keys and corresponding lists of urls as values
    :param test_run: amount of Instagram Images IDs to classify; None or 0 if all
    :param download_process_no: number of download processes
    :param classification_process_no: number of classification processes
    :param output_file: filename of the results file to be written to
    :param progress_report_interval: interval how often to write the current progress into stdout in seconds; if None or 0 => no progress reporting
    :return: None
    """
    ctx = multiprocessing.get_context('spawn')

    maybe_download_and_extract()

    with open(path_to_imgs_pickle, 'rb') as handle:
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
        download_p_list.append(download_p)
        imgid_urls_queue.put((None, None))
        download_p.start()

    result_queue = ctx.Queue()

    classification_p_list = []
    for i in range(1, classification_process_no + 1):
        classification_p = ctx.Process(target=classification_worker, args=(img_filename_queue, result_queue,))
        classification_p_list.append(classification_p)
        classification_p.start()

    save_results_p = ctx.Process(target=save_result_worker, args=(result_queue, output_file,))
    save_results_p.start()

    if progress_report_interval:
        progress_reporting_p = Process(target=progress_reporting_worker, args=(imgid_urls_queue, imgid_urls_queue_orig_len, img_filename_queue, result_queue, progress_report_interval,))
        progress_reporting_p.daemon = True
        progress_reporting_p.start()

    for p in download_p_list:
        p.join()

    print('Download done!', flush=True)

    for p in classification_p_list:
        img_filename_queue.put(None)

    for p in classification_p_list:
        p.join()

    print('Classification done!', flush=True)

    result_queue.put(None)

    save_results_p.join()

    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_imgs_pickle', type=str, required=True)
    parser.add_argument('-o', '--output_json_file', type=str)
    parser.add_argument('-dp', '--download_process_count', type=int, default=2)
    parser.add_argument('-cp', '--classification_process_count', type=int, default=3)
    parser.add_argument('-t', '--test_run', type=int, default=0, help='amount of images to process, None or 0 if all')
    parser.add_argument('-ri', '--progress_report_interval', type=int, default=5, help='in seconds, if None or 0 => no progress reporting')
    args = parser.parse_args()
    if args.output_json_file:
        output_file = args.output_json_file
    else:
        output_file = 'result' + time.strftime("%Y%m%d-%H%M%S") + '.json'

    main(args.path_to_imgs_pickle, args.test_run, args.download_process_count, args.classification_process_count, output_file, args.progress_report_interval)
