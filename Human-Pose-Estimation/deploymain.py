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
import re


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
    print(input_df)
    # base64ImgString = json.loads(input_df)
    # print(base64ImgString)
    oriImg = base64ToCVImg(re.sub('^data:image/.+;base64,', '',str(input_df)))


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



    # Create json-encoded string of the model output

    executionTimeMs = (datetime.datetime.now() - startTime).microseconds / 1000
    # outDict = {"all_peaks": list(chain.from_iterable(all_peaks)) , "subset": list(chain.from_iterable(subset)), "candidate": list(chain.from_iterable(candidate)),
    #            "executionTimeMs": str(executionTimeMs)}
    # outDict = {"all_peaks": np.matrix(all_peaks).tolist() , "subset": np.matrix(subset).tolist(), "candidate": np.matrix(candidate).tolist(),
    #            "executionTimeMs": str(executionTimeMs)}
    all_peaks = np.array(all_peaks).tolist()
    subset = np.array(subset).tolist()
    candidate = np.array(candidate).tolist()
    print(all_peaks[0][0][0], type(all_peaks[0][0][0]))
    print(all_peaks[0][0], type(all_peaks[0][0]))
    # print(all_peaks)
    for x1 in range(len(all_peaks)):
        for x2 in range(len(all_peaks[x1])):
            all_peaks[x1][x2] = list(all_peaks[x1][x2])
            for x3 in range(len(all_peaks[x1][x2])):
                if isinstance(all_peaks[x1][x2][x3], np.generic):
                    all_peaks[x1][x2][x3] = np.asscalar(all_peaks[x1][x2][x3])
                print(x1, x2, x3, type(all_peaks[x1][x2][x3]), all_peaks[x1][x2][x3])
    # outDict = {"all_peaks": np.array(all_peaks).tolist() , "subset": np.array(subset).tolist(), "candidate": np.array(candidate).tolist(),
    #            "executionTimeMs": str(executionTimeMs)}
    # print(type(all_peaks[0][0][0]))
    print("Image predicted to be 'all_peaks: {}, subset: {}, candidate: {}'.".format(all_peaks, subset, candidate))

    outDict = {"all_peaks": all_peaks , "subset": subset, "candidate": candidate,
               "executionTimeMs": str(executionTimeMs)}
    outJsonString = json.dumps(outDict)
    # print("Json-encoded detections: " + outJsonString[:120] + "...")
    # print("DONE.")

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
    input = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAASABIAAD/4QCMRXhpZgAATU0AKgAAAAgABQESAAMAAAABAAEAAAEaAAUAAAABAAAASgEbAAUAAAABAAAAUgEoAAMAAAABAAIAAIdpAAQAAAABAAAAWgAAAAAAAABIAAAAAQAAAEgAAAABAAOgAQADAAAAAQABAACgAgAEAAAAAQAAAGSgAwAEAAAAAQAAAP4AAAAA/+0AOFBob3Rvc2hvcCAzLjAAOEJJTQQEAAAAAAAAOEJJTQQlAAAAAAAQ1B2M2Y8AsgTpgAmY7PhCfv/AABEIAP4AZAMBIgACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2wBDAAICAgICAgMCAgMEAwMDBAUEBAQEBQcFBQUFBQcIBwcHBwcHCAgICAgICAgKCgoKCgoLCwsLCw0NDQ0NDQ0NDQ3/2wBDAQICAgMDAwYDAwYNCQcJDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ3/3QAEAAf/2gAMAwEAAhEDEQA/AP0INxE42xyK3YYYc96hZWZflBP05/lX4seIvG3xA8KfCLXb7w74j1XT7y31fTNs8d27MkczCNwPMLjnceua82+HX7VP7Qdv4y0DSdQ8Z3F/p17qVpZzw31rbTBo7iZI2wyRRyBsNwdxx6V8jgsR9ZoqtsfS5vk7wONeFetj942yrjcCMexr8x/2yYdviLXH7Sadprn/AIBcP/jXR6/+0l8TfC/iDUbJYdO1GC3mdIkmWWEgKeMtG5JNeR/Er4gXXxm8D+MfFus6VFpOo6Ho8KBba5kuYp2V3cu3mRoyDPAUE9+a86OPpV5ezhvc9eHDmMwsXXn8LXc98/4J8am0/wANfGulv0tPEySD1xNp9t/VK+7QhYhR1yP/AK1fnb/wTsmU6N8SLVmOV1LTZQPb7IFJ/Svs34weL4/Afwq8YeMHLf8AEs0W8mTYcMZPLZUCnjB3EYr6JWVkup8Zir+0aZ+eH7Qf7aWpz+JtQ8F/DTVH0HSdGuHt7rW4VDXt9cQNtlSBZEZUgRwULbSZCCRhQGPzxr3xS+Kt/wCFNL+JUniLWl1TTZbmzhvbjy3EkNxgtuj8vY0UqgK6MpDYHsa4D9n34eWnivVV8SeKv9IsNOkUPDIflurgHcxYnlgWyWPcnmv0R8ZX3hnxJ4Tn8Oahp1nY2Eyqu3KxhVUYUKTjkVjWxkKU+Wx6GCyqVWm6j36HNfsX/En4Y/ECDUfDc/gjwtoHjSxh80yaZp0NumpWQIHmKhDlJIy22RAzA5DAgMAPvYuyqFj+RQMKqjaMfTtX4QeC5D8Ev2jPC2oaXeCWwi1W0Xz1bIaxvmNvKhwcEKJAT7gHqBX7wXAAlcLyATjHQjtWuIn7qkup59TDuM3GR518ULi103wXqev3Z1G4NjBsisbHVptG+2TXTpFFFLdwFZIkLsCXBGwZbtXnnwz1TwdB8R9U8Q3fg6z8L65o/hPUG1u089NUlZtPlt54LiO+wWnjnguNyzHDtyGwwYDU/aTWL/hQnjaaeaWBLfTGuC8EayuGhZWX5GKqw3Y3AkDFXrz4Z2+ifE3TtL8KWuyfXvhTrfh58txcS6d/Z0djGeAq7VkmxtABBJPaurByvTuzCaSdjhv2fvBnxJ+PXw0tfi54q8Y+IPD194lvL68WzttTItvs5uHW3METRsIYlhCRqinB2eYfnds+1/8ADN3iX/op/ib/AMGK/wDxqtX9k7TNT0z9nPwHpdzaG2ubLS/s1zDPjzIriGWRJUYHoQ4II7Hivobyr7/nlF+S11uephdH/9D5R+I8LQ/CXxVBKpRmvdKcZHZZxntXx14abyPGXh6cf8s9Z0xvyuoq+4PjDp9xF8PdXiCsAxhLg/7MikHB54NfE2iWF9eeJ9Fs9NgkurybU7JILeBGklml89GCRooLMx29B2yegJr884YqqeCUm+5+q8bUeTNZN7aH3R48habxFqUvOTcuowCxLE4wAOSc8ADr0FfQXhX9krxTcfDfX4fF/iTR/CN14v0xbezsNQSWa4UZJV7hYiPLDq2QoO5erHsPefA/wi0H4aajJ8QfiT5dz4jdvMsNKGJY7Jjz5khwQ9wOw5VO3OSfOfij8VzdX97HcTsJLeBru5C/MYoV5wx6liOTzwBXk4TD+xrzqTet3b7zvzLMqmMwsMJh1ZJLmfyOO/ZC+F3xK+Dnjfx94X8faV9kivbHTrqyvreT7RYX22WdHe3mAXdhQu5WVXXPK4IJ9m/ar0681n9nLx5p+mqZbh9MLLGvLMEYEge5rzzQvjH4lHhRZ4FktYITK3+lskirGg4kDozKI2UAgk5wex4qS9+Lmra/o+p2t5ZQXeh39lE1nqNtcLJHciZTvR0KjY8TrzgspVlOc5A+ooY+Ki3JWPhcTklV1IODvd2Phrwd8Kdcj+G/hzUtONgqyW6Xdz50Alk+c7uCSAOOBjkHnmvVfEXhbWtU07Q7e2vY7WfbHJI7RK/mE4+6HBHbqRx1rfttZOo+GZtFsIVtby1jjj8hhtQjGAVxn5ePzqt4ds/GEkscd4IDBCw2lnd2VV7Lk4Ga8t4yNRupuj62jlsqUVTkrNaWPmn9o/wRqenW/hfVLlorjUnuTbGS2RYslcMrEDAB3hQG45Pav2MsJHm0y1kfO97aEtnrnYM5r4v1vxB8NvEevv4X13VLCJ9OuLdSk80cf7xAszgb8ZwNo69a+qtN8ceEr5I4NO1O1n2qoxFKj4AGB91jivRwtScoWaPkc5pR9taBW+JPgnV/iR4I1PwJpGq2+jtrUa2txc3NoL1PsruPOQRErlmQYU5AB5rkZP2ePiN4p+zah8Q/jX4qk1CO3mgb/hHoLLSooUmUKyW5WBnQfKMtkuQByCM17HpWp2c2oQRRyAszYGM8nr6CvRUC4Gev1/wr2sFUtFxZ8viYuE9T5gs/2P8A4K29uI7pvE97KWd3nl8S6kru8jF3ZhHMq7mdizEDliT3q1/wyL8Df+ffxH/4U+q//JFfTH4UfhXoe2fY57s//9H59+LfgXxvpXhOfS9AvYLzTrqTZt1KQmQJ12Ryn5g2QCu4sCSFGM5H1J+zh8FbD9nzwfa+NfE9hH/wsvXbXzJ2lKSNolrMAVtbfGVWUjBmkHLNwDtVQN3X38PaPfeHdW8S2Ud/Y2mpLOsEiho2niQvAXDcEJKqsB/eUVe1rxBd65eTX91I7vM28g5z1JI59zxnk1+N8M5zPEZapSSUttD9j4sypUsya5m1ZbmB8QvHM9jZTandTh7mVikIdxjeer5OBheoryjwz8Ffi98SNN1G48JeFNS1T+0IZY2vLtPslq8kinnzbjZuU8fNGGWvdvA/jyDw5qEl1f8Ah6yu5Ek329xdQJLcQ8YO0sCFz1xX2P4O+PU/jDS7nSdptr6xQSRMvy74GbGMdMqep969jLsqWLxaqVptJbJfqeXjM7eAy6WHw1JNy3k/0R+R/i/9iD9uDw54B07RrbR9P1fT7W3eGWx8Pays180JwFSaOeO2SVUHQJJ0B6ng6H7Of7MP7T2laLrK6r4C1ex0q48h7O2v5beH985kEjRw+eSq4A3528kHBOTX7JaD4rvmvzeTSMfLU4GcAk+vWvTo9ZW/jBub5ombDBTLETz32tkqDgECvt8TlOHqRdHa/Y+Ew2d4mlUVbdrufiz47+GGtfD+1t4taKwazPIXljgYSiCNcbUdwNpYn72CR71zFnf6vFGzapexmPoqqoUk/wC1iv2U8bTWSxb9UvbYxlSQ920MY2jg8k4645r501fSlicyCO2mVirgiKGRCGyVZSFOQeoxXm0eD6dOPLCZ61TjqvUm6laN2+x/P18bNF1vT/F2tS6nY3CaNrtwl3p9zNbv9nuHRFDxxSsux5FKksqksAc4r5xezg3/ACwKpB6hApHvnHGPXtX9PN7ovh3xbYnw9450ax8Q6NMwEljqNuk0HBBDBGB2sCAQVwQQOa8b8a/8Eyf2cfiDYTX3wz1jWPAOpNDJ5Vm0v9p6X5zcjdDckzKgIwBFKhAJwemPZ/s+VGCimfPVsyVebk+p+X37HnijWV+I+mQXWp3ssSapYR7JruaRAkm9SAruVAJ64HP4V+7Z4O09Rx+VfhVpnww8dfsv/GzVPCXje2gbU9J/s/VLV4ZN1rfW8czeVNC+CwjkAI5AdGBUjjJ+3IP21IbcFtR8LXxy3JtbmCTBJ7CQoSP1ryamIp0K8o1ZWv8A5HtLJsRmFCEsJDmte+qXXzaPv9cY5p2B7V8SwftueADH/pOn6xBIOqNaJKf++onZf1zU3/Dbfw4/59dX/wDAA/41f9pYb+c4Hw1mf/Pl/h/mf//Sw/2ir5Lr4V3KIxDRtvUqSCrCNsEEcgiviHwr+1H8QfC1pFY6xFBr1lAuB5x8m7VUB/5aqGVyBjG5c8csa+jPiZ8UdD8beANR0ay042l2YGm3iUMmI42yABzz71+a17L5luVX+NOPxX/69fjPA+XVKeDdLER1u2ftXH+OisdGdKV00j9aLb4m6KfDo8XakJLOwJi82VwGEYlZVyxU4CruyzHoATX1L8IYJ5NX1DUdqpCLaOGJy6ZlMr7iyICW2bQPmxg54J5x+dVzb2Wq/sy+IpRNH58OltP5ZYbtmxGB29T0xXnn7Dmt3Xh/xvrWqW291t00cSqGLf6NJcyQumCeFw4wBwOMV9Tk9XkVevU+xKyXlex81nWF9qqFGDt7SF36pXP340WTaFhChjJIqgZC5weg3FRkn1Ne0aNNd6k5hguQ7QkebaX8bIw7ZRiqumfq6+hr5b0rW5pfEc2jmAS/Z0iDowyrSSBpWyP9kBQK9ctPG95p8C2+HKxcRiU7yg9Ax5I+tfbYe05OaZ+XvRNSZ2vjX4Z6P4uh8vVoHRlilgQLKy7FlwW2Sx4OTtHLLxXjjfBTRdH1C21WwS5t7mBuZN6yedGIxH5UjkFinG7GR85zXcxfETUJnAkZjn34rSPi4un7/pnIxivVhGxyOR57P4dMDiRlAPH51o2zeQuCeAMVrX3iOzlB3ZJ7V5tr/izRdLkhfVLtLOGZjtY5LNs5IVRyWx2rDEV6cF72x2YSjKpL3dz5P/4KA+DHn8KeCfiZBGpuNIv5tHvpixJFnqYDRL6HFzFEPxwPQ/mnNGJIt6j73PFfp7+098adD8SfCnxB4OtfD3m6ZdwR4vrqfbdRXMcgaGeGFVKK0coVhubOO2eD+ZdjmexVwOuc/ga/NuJ8RTnNTou9j9r4Fw1SEZU6i3VzANqM9B+QNJ9lHoPyFbzW4z0pv2cen6V8h7WB+h/V12P/0/y+8JX8954mlt5JC0P9l32UycFgq4OPXk149IT9niPog/lXpXguG6g8WRtNFJEJLW7UFiMEmMHsT6V5rMuLZPZFB/LmvmqEIq1kfSV68qydSb6n0nJqSw+ALXTWYD7TosKnPVgydPpUn7EatL8bLSG4LDT1tm1DUMZI+z6Ypn5UD5szNEuP9rPauB1G6Y+H/D3zYDaOoI7HbxX6H/8ABNT4QT2Hhnxh8b9RgSI3ijw9o7yDDGGKTzL+SM55Bk8uHOOsTHuMRg8InGsntJmuZ5jy+wqL7MbfhY+0PCMmrz+KNP1a9jaL+0zc3uTwWhAEQODzjOVH517jOT50cSx+bI5wVHpXnUMkR8eA+buki0xcr/CFMhYHHYcV6vYwsls1yoBuJx8p6mKI9z7t/KvqMFTShsfE15JyuislvFcXYhtEPHHPb1rTu7SG1j8otvfvVuwiFhbvdzADcOG9TWMbh7hmkeu/ocrj2OfuYGD5HQmvi/4xfEDTtT+Ktn8OrJt1x4ft1vr0/wAKveBhHH7nYhZvTK+tfcr43DPPI+lfl38WvCj+CP2t9e1eRiLfxtplrrFqTkjdaotncIpxj5GSMnnJ346CvnM3TdKVj6vJLRnF21GfHDcvgS+I6Hyce3zivmHwwpl0dT/tMD+dfUnxli+2fDPWbiI7/ItxNgekZDHP4CvmH4dGHUtEdo3DGOZmIHOFboa/L80fLhpSXf8AQ/Z+G6nNjIq+6NU2QPaj7Evof1rrls4wME077JF618l9Z8z9L+rx7H//1PzwutNca/aaraGOW0ia4jaRSqEh4mGdp+bOe3pzXzlcqAm1OR0H8q+zdT0HR0uGkN7f7Gbo1oqqDjGOK8B1L4YandyXEuiXdtLbRPy145gl+b/ZWNhj07183Tep7PtPc5CqLLVdV0PwfYaLB9r1C8hWws7cdZrqeZIbeM5K/wCskYLnIA9RX9FekeA7X4NfCTwT8FfDwHm6RZw2906qAZLiU+bdzMBxueUyOx7k5r82P2Avg7Dr3xO07xv4pVJdI+FlhJfOqndFPrN6zx2caEhS2xBI5yAMlCM4OP1W8XXc62Ws+ONSQRNHbzG2jJzsUIeSfXHFd2Gpvka8zhzKspJR7I+fNb+IPg3wV4hvfEvibUrKwjuxFZ2puriOFXERbao3EE5YktjgYGa0IP2mPgPpUi6h4g8d6FA13ErLH/aEWHPcHDErjpX4DftA/EOb4mfEW4uxObjTtNjSyswX3xjYMyuin5QXc4Yj720c4xXjkdvDH8yIEPTKgD+VetSr8itY8Wy6n9MUn7W/7P8ArI23HxG8LwJEQFQX6IPwyefrV5P2k/2dljLf8LO8JlfbV7fI/Ddmv5kJJMDbkn61UEUJO4ImfXaOv5Vuq91doLR6H9RnhX46fBHx/qX9i+CfG+kavf5P+jw3KmQ46lRwSPQ4xXyz+2b4k0rSPHnwtspoU+23MWtPHdDqsANqGjHs7lW+qmvwggvruyvYb+ymkgurdt8U0TtHKjDoVkUq6n3Ug12x8c+LdY1nTdZ8Sa3qOr3GnMqwSahdy3bRREjKRmRiVBxk+vfNeZioqpFo9vLcSoVI3P2X0qwt9Z8O6jZXrCSK+tpYipGeHUrX5r6boviOxa61HQI54hpUpt5ZbfLMpGR88YJJQ47A47ivvH4YeJDrXheC7jOSV9fUV454EtvsfjjxtpifeS7jmHsJdzfrmvzfMcSsHTqTcbpK7TP1bJMJ9erU4Rm4u9rr7zyfTfi1rVtarDe6ZHeyqSDNE3lhvqOefWr/APwuG/8A+gD/AORT/hX0Dd/Djwpq8xvr7TInnf7zLlNx9SFIGfeq3/CpfBP/AEC0/wC+3/8Aiq+UWfZG9XQ/H/gH6Ksg4hWkMRden/BP/9XwC5k02VY2M9wzucOiru2IB8xIJyRnjjPviuF1HTdMhaW6jvLgCfB2z2xjUbeMdTn6ivcDYWtvbytd6lZrPIXMTRN58iocqoKgcyYIB496qzpYH7LB58E0ZUojzMEX5Rk44bbj+InBzXzCvuj0rs+8v2DfBPh/T/gU/i1Z2upNd13UJpm2FEc2L/Y41G45wixkccbixHWrH7avjhvBvwG8YapbSrb3H2CS3tiTj99cYijAx1JZgAO5ruP2TrX+zfgNp1nCq4Go6q+2Jg6qZ7hpDgjj7xz9TXwr/wAFOvGqad8PdL8HQuPtGranbvKpPzBLUmfnHoUFevhfgSPMrO8nc/DnyxCFjU5AAUH2HAqRZAVIz0pkhGcA5wOPwqOMfKT2rssrI5uVDDJk0I/BxUDYLYFN6e1TKTat0Goolzl6sQy4cD3Aqop+bmkPLH61nJIa0fMtz9JP2XPEp1DTL/RJny0BUoPVWUEV3ugQrB8bPFkPQ3Wn2U2P90lSf/Hq+Y/2TNQI8aS2wPE1sNw9CjY/rX1SbWTS/j5I0hzFqWhbkPqYnGR74P8AOvzniuj7lVLrE/ZuBK/7ylK+0keupANvpT/s49a1IYQybumal+zj1r8D5vM/onlR/9bxbUdN0lI2MU8EU0SKXcswYSBuoO4kFv4t3UdhRBaafJHHCsMl7EWDzSMm5Xyf4BnGzthOO+KmudY8WtapB9si+aUW7GJUkZwB0ZQRgAnGDnHXiqc2meIrOZYbXWG3yxecuFkUqDx94SKM/Tp6GvlFJrY9upTtqj9DP2d/GFv4Q+Bes6skTxxWes3UFnbSEKzSzxrIW2j+BT3r8Sf2zvifd/EP4ttZG4a5h0CHyWZh1ubnbJNg9wqiNR+P0r6N13xj4k8N6RqWqalqt2tvpds8wdCzI8ka4VT+9Kb26DcuD29K/MrVNUvNb1rUNZ1GQy3d9NJczyEDLSzOWbpx3wOnA6V7WEq3p2seFi4u7Mwjn8MU+P8A1RppxninJ/q2rslsjnjsUB980jNyopCcMcUg6gntUFD/AOIUucHNJjvRUyVyd2fVf7Itut/8UXsmnWAtYSTIW6N5TpuUe/zg/TPpX3f8QtG1nTvih4O1eeLdYPBfWAnjwQssypIFfuMiMlfxr8lPAetXug+LtM1Oxvl06RLiOJ7lzhI4p2EUjOcj5VRixzxxntX7NeI/F13psM2jeIbeBrjR5fJlmU5BeIhMr7kHivk+J8NL2Eprqmj9L4GxS9pGD0s7m1Cz+WuASMVLmT+6ap6Z4y8Hz2iSPcGJuhV1OR+VX/8AhLPBv/P4P++DX82PA17/AAs/p9Yqk1e5/9fyJNP1+Lz7k3scf78ssbxNLKA4AIHCLjjuxNQo9qiyWdzrcdoyuAzLaM+0dcMpk+f0PeotW0nw+JWh8pvNgXz9gQMhU9F/ukHuOWzVKbwnoWtPHdTrBHFGoVUKrgu2MjsRg++a+TPpeliPWtK8J+JNHvfD2p6kr2N5uF3HFaNFG6jlXUCYEOvXOc18UfEn4I2nhWwk1vwVrE+uWzPGPss1p5d2N528FHcSBTzyoO3JJ4r7XHgvRxJLDbiONTtwzEEq6E/MAexHHU0DQ7c2qRyJEY4mceciL5hTOd3ygc5GOa68NXcfQ8/EUYyex8C6Z8Bvifrdta3Wg6WupJeQJPGsUyxSAOAdrJLsKsvQgmtWD9n34iQ2rf2np09lMrFWhMRkK49WQlTmv0x8CrBa6vDJG0u1o+POwXJ9eO2BxXtk6xtlwMZPPvVPMpKXJI6FlFKcFOL3PwK1PwX4q0uaSO80XUYwjldzWU6q2D2Pl4Oao6b4W13VboWtvaSIQcO0oMYT/eBGf0r97dYUnT5EU9VNfB2naBaS+OdWkmCnN85xjrnGPyrV5graERyW8m76Hx3F8GPiRdShNP0hrqEnAuFYeUff+9x9KseJvgn488K6I+v6nap9lhH+kMpKiI8Y5YDfnPbpX66+GrdIoQAoAUDGOM/lXz3+1vq0cHw3bTwPnvLy3ix22hgzfopqFjpuSRc8ogoto/Ovwt4Rm8QavpGniCaaHUL6K3kkT5UWIMPPy2OMRbj14xX0/qnxMg1/VtY0/QpFt9Dh1grpyl3mlubZDhWMjklvu5zk5HNeZadqlz4X+FJ1SOZo5dUjksLJVJH7yZ2M83HeOJePUnFcr4EMbxTKUGbSRDH/ALO4YNVjv3lB3QZTWeErxUd2fTMfiqaNdokNSf8ACWz/APPQ/rXkcmpBXILc96b/AGmP71fFPKovWx+ox4nq2P/Q8dbVfCNy51J4JVIjMSbY7ZGPGSGj3nCknvznvWE914SvbTybwz2RNw7SRr5CqyIcBxtzjJAI74rRuJr37f5gcz2eOHgASL5iCwBb7+AO34VSivrLzpkntWicsWdpSsQUMSQAcvvYgAgcHBr42svePfptuzY+81rwbHMba2F5JaorPGsskbhsY4BRN+Ae/pUL+NfCxiZdPLLKyhebgs65wG2Yhbg88nBqCb+zfID3LCC3l3eWnn7pHULvc4UDbkZG3NWrBvDjzQw29nBbokWYw6kSbjkKuCBj6ls5NNOyNJwi0zb8JeL9I1HVVtbUNGyTHyi5JYov8JJVcnPXivpRZlkQEMenQ1852VroNndQ/Y0jjnLsPPm3AGQjeQqgsN2AcngDvXsuhXjzQA7t49c5rmxtX96pI9PLoc1O3Y29SP8AohB6FTXx6tstt401FgPlM7Efjivr7WJ44bVi/Vlr5TnIk8VTuP4pCT9RR7ZtHfKhoe8aTIIbYFu4GMfSvlr9oGws/GFzYaDc3M1ukIlux5BTJcLtUNuB45J45zX0M2ox2lmC/p/Svk/xdq41LxleAuY4hCsaOyjGF5OM9ea6MFeVVXOHMZKlhmu58m+LbR9MtrHS47m4ntLcM0MUrgrE7nL7QowATVzwEw86/HAysbY/Ej+lW/HlkIWjlRQI97DIzyc5PHQfhWX4DONQu489YM/98tx/OvelH9y7Hx9Gq/rMUdpOhMzcjg4qHyz6irdyNs7gDPPWoef7v614nKz6n2vmf//R8Om0WUlp9ZmDxOd5U6hBIr9CAAG2cZ+6BzWJd6cyalJNHHAlhuEQLXKsqKRwNoXAbPTBr2TW9COkaakMvnXbsFbzkwkKsyDDOqooUdg2CSeBmuFk06O+36HZT3H2swGadZJCqvIOhClsc9jxxXzdaC3PUhXdtDlUms7hGTTp4FuE3FFvGkG0gcnCxY/Iis020saJdT3tvdmWcFIYEunVivc/uwhGemMY681uNplpNqo0syOdRSHz/Jk/1DiMYdA2Ww/IOMjdnIPBqKbR7eZI5JoZbJ4XICtgpNuB+X+LIB9DniuV2SudMKjludHbOkQLXNymnoV+aSBWkj3AHoC6uHOf4gQe9etQRXumxRXURDQHa4PTKnnpz6+teI2HhXT5L2CC/ZpPNYMgChSS5CgALjhc/WvqHU9FlSzjtYV2xxRiML7KAB/KvPxUuZJI+gyuGtjkvFmqRSaV9oh4O3PHqa+dmhZNTjlH3nG4165rlvPHB9mIO0Hp+NclqWmrFqm2IHcFjIBGCNy5I/CsIUnozvqTTnyvcd4klDeHyIMmVwF+XqPeuH/4V74A1HS01PWbvWY764XDxBXSErn/AJZOIyVHcnOMcivQ3srgwyyKpx5Z69uK9H8OR/2t4T0wWso+2mxUxwqqM2yMeU24OVzyDtOeDjtxXtZXFe0bPB4gny0F6nyTrPwm+Gl0oFwmqNCxVhG5nYjBBJR02ZBHFcPN8Mfh0zDT/DSXmhXsyk/bZ3uJjHECCQYppBGVfpnIK9eelfaUdlc2ytb3+UbTpVhVWy6yRYzt5BK59vzrh9Sihi1SGysbb7OZY/OkPyl9m7BL9wvAAB619CopXXc+Ov1R8lXHwN8QPPJ9i8SExA4DTW4Rm4zkAt054qL/AIUV4r/6GOP/AL9L/wDFV9SarY3Ul2XhuPLUjIGF/qDWb/Z1/wD8/h/Jf/iawdGJ0LETtuf/0qd1p/8AZwil13UAJY0Bk2XBiWOIMdshjz8pIGNxyD2rK85tQlkl06aGS2eMOs0RWVnXt8o5Py85robz4S+GtaiuEufFuqXyyEwym3sXwwcksoYRq2w99pC1lzfCvwxFFfym78U3Jt/lmit2jRpflULsUzRsylcDIAAx3NeRUw05HQqqWx5je3eo/bJ5mF1byeWE88xod3lsfK2qSAOGJ+bk9qhuby4urRJbqdoSZw8bqyMgIUDoPu5Pv1zXV6/4K+GOhhbK2tNa1CWTAdLjUDCox77m6dvSuP8A7M+G2kmVZvA8VwP9WhOpKUcjswAJOCcjPvXnVcJJ6NnbRrxtax1nh2zsv7e0o28kkvmTowZ3D8plnxySAT9a+uBbR3MeCAwPU18peGrjwimpWsOj+FdI026i4W4gu5J5owfvbQUCj3FfUvh+Yy2pLHtnNeZWg410j6LLaidJ2PA/GbrZ6kkahUUTgMWHy7e+R3A9K5HUrjTrzWIrmxOEZSrfKVyV4JGQDj09q63xVq8uleJk1OIRO8UjhVni86M5GOV71zFxq+o+Ir9tV1WGGCUHy1ihjEaKi9MAAdzmujl/ct+ZDcpYpa7I7COyhfQ7mcDJC4H41xXh6W8vfDkWnwabdSeQk0KyoDtmV5CzJ8rbtpxhuma7d7gQeE5+MHdjP0rhPCXiHxrY6UtjAs8VhIC9pIs/kqyO7FxlVJJ3ZOCRjjiunK1+9PPz6pfCJPuR6novjG/lke10C+Ic7tjcBSCeY8ONjYPXnI69Ko3/AIU8d3TWYi0KSAQOCjzzRq2D94ED73Ge4yeeMVu6g/ia4hRb26eyikdNsj3sqgMCeTk4YkHn0rPudP8AEGpg6bq9nDGscmYHkuJrkylF+WRXD8ZXt619HzrY+SclcwLj4b/EaaQyRWX2qNslHeZCwXJwuVZQcdM4qD/hWfxJ/wCgSn/f0f8Axyunb+1IcJdR2MzgfeWKSTAHABK8ZHcdab5t3/z7Wf8A4Dy1haQvaPsf/9PpofBsMIaT7TPLKs4MohnmRmJGTG5J3CMnlgDjFYV1oFmkk8UhunuWRJEWe7GxnYfKA5yxTJwvTpXVmC9ivv7NLXCQXLsski3AaRiDkMhA+U44J7r1qrrL2dukTWojit2JVpHHzh0wpBV8b8MMjH5Vzkc2tjwifRTFHJbagonkmkM7THcUgx13McdO23g9a56ysdKtrtrm9MKxbcBR+8G7tux0OeSemDXSa54jmjfULeaGUCOaZI2DGMyoxA3yB1wMegrnLVbnVbgXMDiVUjVxG2zO91ZVkyucjrxwcVwVfisd1E9FsbHRgbbW7BRb3Im2TRgFA2R0CjrwQ3ToQa+ivC8oltjs6bTXytox8TWUcQ8TXGm3e58wrbk28isvG0qcq7bCD24Br6a8EOXtC3+w2frivExsLYiPmfRZXL920eF+L5oo9di88jZ5zjaQSTwTgY6HOMZrKmmjmvHEBZ1DfePcnng/pWl4jAm1ucSc48x0PAZSO6571y9lbW9oUW2nlmB+Z/MPKMeSvBOfWnf/AGdvzN4f74l5HXa3dCDweyjO5yzD0OB61l6OdTttAtrJIIJoYljciFwCGmI+VySAGPf1qbxeoj0GyhLeWHXk4zwTz/n0rlbXxH4fXR105bQvPqCRbmy7Rme3c/eUqSFBAcKDz0rtyum1Uk30PKz3SnGD66nSarLBdanZzRzmGe1hkSwiSNXVnUcglxy3oOpHSsy5j1/V9Ua7V4IdMhaEyvGsrTROibnLAbQmQQQwLADvWrY+ONIcTaZPHdPdxx/u3jsmCOAQdw3YCEtngHIHeopPEc32l4tP0e8+z3fzSsAocsvA7kMNowc9eM9K9+HKtWfKypu51NvPdDzHhUSJJIWBCcDoOPm74z+NWftN9/zx/wDHP/sq5fT9W8Q2cLQ22mt5G9jErkb0Q9FbAxn6etX/APhIPE//AEDf/Hv/ALGn7SPY05Ef/9TtJdK8fXij7JoVrDJAxaKbfg5AwP4RyPauS/4Qf4v6rb3Vtqd5YtHNtBLh5HRQS3AYgK2f4lAr1u4+Imq3wcWl1a+bJEI44oUeVUZTy+Tksf4dpOPTFYmp/EDxHKZXkuba2gnTYsUWntLKAo5wSclj6AkenNefz1eiJvE8U1n4H/EXVbw3Wo6/bDJypih3MMYyWJbknHOamh+B+tKIHn8Sy5hUqrW0MaMq+h4OR/KvRF8TeKL2VdUsbq6u1uIQyRw20UKlJcbRsPG4Y53HcO+KxPEOq62yTwz2mpeZNKiwvFeJbKFGC5YI2NwIOecYrKVGb1ZaqW6mAvwQtgVa/wBe1GYBvMUFlUB+QxHyjBIJ6dq9Z8D7bbFkp3LEzwhj3C8Z/GvgH4iXOuvqQvbl9VRJZDbmynunmVnD4DAFioyBgEcEV9h/ASO5TR4LedSjJvARsttyc4z6815WPpW5Z9Uz6LIan7xxfUwNT0PwhreuaqviiFbg28qrEhk2Y4JJFcjJc+CLvVls/BTRj7LmO8SJt+2UkdT9K6n4haD/AGR4xv8AU5nEZ2q8YCl/vLgll4GB65rxf4GWiahrfiW9luILg/a41HkKVG8Ju6Y7qR+NY1oOOHU11OrD1L41x7HtesX+n2Wu6V/agU21pFkq5AUyMpVASfdq6NPFHhNYfJS085oGzI8Vt+7QjPJkYBT+FeW+IILjWfEzrbKlxHEyRnLKqDJwTkg5YAHA9qzZrRnv/wDhGzHBKxkZo988skceFyfNVCEQ98Mc9sV7WU0uejzs8XiDEXxPL0SPcP8AhKtKtLdnuoGsbd8YlaJWLZwfkUcjI71if8LH0HUZZ10KMSxxhkaeaRIcMuMBAVJII6t61ytr4R0vUbeUzLMybfKnkSR/NDoACyZJIByOAQPrU8XhvwdPY+Sm+SKTbbFJflk81T1IwpOSvrzXoyoa6HiqtobjeNEDNujteScF5QWIHGT8w7j0H0o/4TaH+5Z/9/B/8VWLPrniLTZPISC2uoG/eQSi2eQNE/I+ZI2HByOpqH/hLPEX/Phb/wDgHL/8ZqfYEe1P/9X1rFxd6du1MHQLOFQrjzAS0QXCZeNsgHuAck85xXMLo8BsIptH8yJ78ZF7MhNxsIOVQyF2DMFBwqDaO4p/iHxXqlnrcGnQaauoWjRGNIjFuWS58wIhaVhsUAHcVznGasagNd1W8h0rU57dGCmU2liGcsqN93zAEbB6lQcdByK5znKujxQ26IIomeOyDIQwWJ5PMIzKUBD5POd+C2d23GDWb4g0jT74trU05kbZHHPaEKkLgkglWIPOCAyjbyOtaq6rplrLJK1uYrlpeYYv3kruOU4AJyuDweF4HSq3iSytxYvNE8dol2khuUO0SMWPHyOQAwyc49elDA+RvHul38l0q2dotlvkJkBYTzKI2yqoQdqgj5vUDivtv9nLQm8Qaek0jtHHZxGW4cbVZpHbCRjgjcwGSMEqOcV8D+P9eFteLBdTwJELsNLl41OwDapVVkJBP8R7jjFfqn+ylq2lXPhrw9oMmgtbHXNObWtNvxGzw38ERwtw06oYw0gcoI2PmbQGII6eXWw6qzSlsezgqtWmpOitbHEftHeB7TWvh/qHi74fWjQ6zoPmnUUumkC/YYlPmSxk7o2ePG4EY3Dg44riv2OPDWrL8N9F8VWMdtqNxqqS6ijT4WSJLhR5UEbOCy5jw2XyDk4xiv1AvNLtb/S7izmtPtMcitBKphd4mVlKsGBUqwwcFe/c4r86/hRoOnfCn4leI/BFlcXVvY2a2Vutg7bobTZ5rhLY4/1O2QCIHkIApPAr0lSp02rLQwoVHVjOU52drnHfHT4exW+p/wDCVWtk9tLekwTxquFFwPmQsFACH7yllyDXh+n+DbKLUp5NUjMYkORJK6qlxJhTllQ84zgFvmI7V+rPiCz8OeINDuoNVR3hmQBvtEgQkjlGBOCGQ8jpzX5Ma23j7RfF+o6Ba6bHqVnY3rhb5SsG9TtIbauV3kdexOa64zpWtHQ4ayc7HUatcTaVO6W9hPJeXylbfMznb5fXYrsVCtnBCjJ6dqh0mLU9Qkt57+3No8cGWgiz5jSA7SzKh2J/eU7jgHFZFlp/jV43t7HT4LSOR/M3GcsxJzx1IA9lAFaGneE/HJv4ik9taytEVkaMFiwB4Dc84zQ6sF1MI0m3Y6KeTxDaOIYI4GQKCBJBFIyf7OS46fSoftvif/nhaf8AgLD/APHKpy/Dbxa00slz4mnjd3LbUA2gHsOD/Om/8K18S/8AQ03X5D/4msfrcDf6nM//1vS7f4W/ExbnzzrttYja+4JEjDMnLsPcn1zW0nwj8RzBY5PFL2ybAT5a7ctnlgM9foa1fFXjzRPDc7SvNfmUxkr5VtEA3oCzTkgZ64Bo0L4uxanpSaxJFfSwq32Z1MsUbmQcE8Rtlc9DkHFeb7VvYqVA59/2fbG+CT3XiK9uPJOSc+W5b1Dck/lUdx+zh4AtLZpbua8ui7b3NzcMQD7F+g+lbc/xDnvNY/sWw0ZJpLSylvLi6u9SuVwAQoEUMQ2FiTkliAAMAVkQfEm71OGFrDS7FYzebJ0ukMwESKfubicsSP4vWp5qjH7FJXON1P4BfCi1Z7+10SxlLbcTSMm4t2bJGSa+g/h98arL4VaZb+BdVMVto9hbqdPmUO0ItVGChcKVRkPBBIwvSvGIPG+vadfSSre7Y9QlMPkx2sKxxkcqYx/BgcHAy3c1Y1D4i3VwJNI1C4v5CFbCxyhIWDL8yMv90j1zSjTlzbjdTkiz6h1T9tj4U6bDEt/4s0ezifCxwRMJJzuHQDJJJPoM18XfHCaw+PmoHxj8LfE17o906RWl5qton2Zp0hZ/Kjfz1Aby97YIx15aqFv4XliAFolra2sm6MRRqC4OC28SbAytgEY5FC6VqK6TaC6u0Eckrx7IovmI2/xyAqWJ6kgDntXZyO25zKrc+aL79nX4hXN4l5r/AMWvEDvkBZRfEwhQMgsVZkOO/wAp/GvpDw1oGneG9HstFl1mfWJURRLeXEMk885AOZH2qgUd89axItPtPDNxDaaTcXSLPIIPLdg8Y25djg9AemK7WfSNfnWOSO7jEJ8v92TIvyvnf93Azg8cfjSUG92Ep2M+68SeFIfPt9NulvZbcnzBbgRqgALEsZG3AAdTg81zFn8RoxJMjwmNoHKFJJkQkAAttbac9sGtZvDFtHMPPSGcW0pgiLxjMKsgx5ffn+IE9e5rQk0FLbQbi9kjt5RI8SLKYgJk81goYHHUMc9e1L2CuSq3Urab8QPD2oRySRpJCEk2FJLiRjnYrHkRgEfNxjitL/hMdB/vH/v9L/8AEVTUReH5bjSoGe58maTzJZ1Qu8hPzHoeM9PQVJ/bLf8APNP++E/+Jrb2AfWGf//Z"
    result = run(input)
    # logger.log("Result", result)
    print(result)
