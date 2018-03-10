from __future__ import print_function
# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

# My own Library
import cntk as C
import sys
from cntk import load_model, combine, CloneMethod
from cntk.layers import placeholder
from cntk.logging.graph import find_by_name
import cv2 as cv
import numpy as np
import math
import time
import util
from numpy import ma
from scipy.ndimage.filters import gaussian_filter
from download_model import download_model
import base64
import datetime
import json
from itertools import chain


# Global Value

####
## Running Parameters
###

scale_search = [1.0]
boxsize = 368
stride = 8
padValue = 128
thre1 = 0.1
thre2 = 0.05

### Trained model..
pred_net = None

def clone_model(base_model, from_node_names, to_node_names, clone_method):
    from_nodes = [find_by_name(base_model, node_name) for node_name in from_node_names]
    if None in from_nodes:
        print("Error: could not find all specified 'from_nodes' in clone. Looking for {}, found {}"
              .format(from_node_names, from_nodes))
    to_nodes = [find_by_name(base_model, node_name) for node_name in to_node_names]
    if None in to_nodes:
        print("Error: could not find all specified 'to_nodes' in clone. Looking for {}, found {}"
              .format(to_node_names, to_nodes))
    input_placeholders = dict(zip(from_nodes, [placeholder() for x in from_nodes]))
    cloned_net = combine(to_nodes).clone(clone_method, input_placeholders)
    return cloned_net

# API initialization method
def init():
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')

    # Load model using appropriate library and function
    # global model
    # model = model_load_function(local_path)
    # model = 42
    try:
        print("Executing init() method...")
        print("Python version: " + str(sys.version) + ", CNTK version: " + C.__version__)
        print("Start downloading model...")
        model_path = download_model()

        base_model = load_model(model_path)
        data = C.input_variable(shape=(3, C.FreeDimension, C.FreeDimension), name="data")

        predictor = clone_model(base_model, ['data'], ["Mconv7_stage6_L1", "Mconv7_stage6_L2"], CloneMethod.freeze)
        global pred_net
        pred_net = predictor(data)
    except Exception as e:
        print("Exception in init:")
        print(str(e))



## Convert base64 image to opencv
def base64ToCVImg(base64ImgString):
    imgData = base64.b64decode(base64ImgString)
    nparr = np.fromstring(imgData, np.uint8)
    org_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return org_img


################
# API run() and
# init() methods
################
# API call entry point
def run(input_df):
    # import json
    
    # Predict using appropriate functions
    # prediction = model.predict(input_df)

    # prediction = "%s %d" % (str(input_df), model)
    # return json.dumps(str(prediction))
    startTime = datetime.datetime.now()
    Mconv7_stage6_L1 = pred_net.outputs[0]
    Mconv7_stage6_L2 = pred_net.outputs[1]
    print("Python version: " + str(sys.version) + ", CNTK version: " + C.__version__)

    print(str(input_df))

    # convert input back to image and save to disk
    base64ImgString = json.loads(str(input_df))['image base64 string']
    print(base64ImgString)
    oriImg = base64ToCVImg(base64ImgString)


    print("oriImg shape: ", oriImg.shape)
    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
    print("multiplier: ", multiplier)


    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    print("heatmap_avg: ", heatmap_avg.shape, "paf_avg: ", paf_avg.shape)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        print("imageToTest: ", imageToTest.shape)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
        print("imageToTest_padded: ", imageToTest_padded.shape, "pad: ", pad)
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        start_time = time.time()
        output = pred_net.eval({pred_net.arguments[0]: [im]})
        print("Mconv7_stage6_L1: ", output[Mconv7_stage6_L1].shape, "Mconv7_stage6_L2: ", output[Mconv7_stage6_L2].shape)
        # print output[Mconv7_stage6_L2]
        print('At scale %.2f, The CNN took %.2f ms.' % (scale_search[m], 1000 * (time.time() - start_time)))

        # extract outputs, resize, and remove padding
        # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
        heatmap = np.transpose(np.squeeze(output[Mconv7_stage6_L2]), (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
        paf = np.transpose(np.squeeze(output[Mconv7_stage6_L1]), (1, 2, 0))  # output 0 is PAFs
        paf = cv.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)


    U = paf_avg[:, :, 16] * -1
    V = paf_avg[:, :, 17]
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
    M = np.zeros(U.shape, dtype='bool')
    M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)



    all_peaks = []
    peak_counter = 0

    for part in range(19 - 1):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
              [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
              [55, 56], [37, 38], [45, 46]]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print("found = 2")
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)


    print("Image predicted to be 'all_peaks: {}, subset: {}, candidate: {}'.".format(all_peaks, subset, candidate))

    # Create json-encoded string of the model output

    executionTimeMs = (datetime.datetime.now() - startTime).microseconds / 1000
    # outDict = {"all_peaks": list(chain.from_iterable(all_peaks)) , "subset": list(chain.from_iterable(subset)), "candidate": list(chain.from_iterable(candidate)),
    #            "executionTimeMs": str(executionTimeMs)}
    # outDict = {"all_peaks": np.matrix(all_peaks).tolist() , "subset": np.matrix(subset).tolist(), "candidate": np.matrix(candidate).tolist(),
    #            "executionTimeMs": str(executionTimeMs)}
    outDict = {"all_peaks": np.array(all_peaks).tolist() , "subset": np.array(subset).tolist(), "candidate": np.array(candidate).tolist(),
               "executionTimeMs": str(executionTimeMs)}
    outJsonString = json.dumps(outDict)
    print("Json-encoded detections: " + outJsonString[:120] + "...")
    print("DONE.")

    return(str(outJsonString))
    #
    # except Exception as e:
    #     return(str(e))

def generate_api_schema():
    import os
    print("create schema")
    sample_input = "sample data text"
    inputs = {"input_df": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/schema.json", run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    if args.generate:
        generate_api_schema()

    init()
    input = '{"image base64 string": "/9j/4AAQSkZJRgABAQAASABIAAD/4QE0RXhpZgAATU0AKgAAAAgABwESAAMAAAABAAEAAAEaAAUAAAABAAAAYgEbAAUAAAABAAAAagEoAAMAAAABAAIAAAExAAIAAAALAAAAcgEyAAIAAAAUAAAAfodpAAQAAAABAAAAkgAAAAAAAABIAAAAAQAAAEgAAAABUGhvdG9zIDMuMAAAMjAxODowMjoyOCAxNDozNjo0NQAACZAAAAcAAAAEMDIyMZADAAIAAAAUAAABBJAEAAIAAAAUAAABGJEBAAcAAAAEAQIDAKAAAAcAAAAEMDEwMKABAAMAAAABAAEAAKACAAQAAAABAAAAPKADAAQAAAABAAAAlKQGAAMAAAABAAAAAAAAAAAyMDE4OjAyOjI4IDE0OjM2OjQ1ADIwMTg6MDI6MjggMTQ6MzY6NDUA/+EKEmh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8APD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNS40LjAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1wOkNyZWF0ZURhdGU9IjIwMTgtMDItMjhUMTQ6MzY6NDUiIHhtcDpNb2RpZnlEYXRlPSIyMDE4LTAyLTI4VDE0OjM2OjQ1IiB4bXA6Q3JlYXRvclRvb2w9IlBob3RvcyAzLjAiIHBob3Rvc2hvcDpEYXRlQ3JlYXRlZD0iMjAxOC0wMi0yOFQxNDozNjo0NSIvPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg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"}'
    result = run(input)
    # logger.log("Result", result)
    print(result)
