import sys, json, time, requests, os, cv2, base64, numpy as np, io, re, math
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib


def pilImread(imgPath):
    pilImg = Image.open(imgPath).convert('RGB')
    return pilImg

def cvImgToBase64(img_path):
    with open(img_path, "rb") as imageFile:
        base64_str = base64.b64encode(imageFile.read())
    # print(str(base64_str).rstrip()
    return base64_str

def pilImgToBase64(pilImg):
    pilImg = pilImg.convert('RGB') #not sure this is necessary
    imgio = io.BytesIO()
    pilImg.save(imgio, 'PNG')
    imgio.seek(0)
    dataimg = base64.b64encode(imgio.read())
    return dataimg.decode('utf-8')

## Convert base64 image to opencv
def base64ToCVImg(base64ImgString):
    imgData = base64.b64decode(base64ImgString)
    nparr = np.fromstring(imgData, np.uint8)
    org_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return org_img

####################################
# Parameters
####################################
# imgPath = os.path.join(os.path.dirname(__file__), "sample", "ski.jpg")
imgPath = "/Users/hzzone/Downloads/4.jpg"

### test
# test_img = base64ToCVImg(cvImgToBase64(imgPath))
# test_img = base64ToCVImg(pilImgToBase64(pilImread(imgPath)))
# cv2.imshow("", test_img)
# cv2.waitKey()
# exit()

# --- Edit and uncomment when calling a locally-deployed Rest API ---
# cluster_scoring_url = "http://127.0.0.1:32773/score."
#service_key         = None  # Set to None if it is local deployment

# url = 'http://<service ip address>:80/api/v1/service/<service name>/score'
# api_key = 'your service key' 
# --- Edit and uncomment when calling a cloud-deployed Rest API ---
cluster_scoring_url = "http://40.84.47.0/api/v1/service/imapp3/score"
service_key         = ""


####################################
# Main
####################################

# Check if scoring url and service key are defined
try:
    cluster_scoring_url, service_key
except:
    print("ERROR: need to set 'cluster_scoring_url' and 'service_key' variables.")
    exit()

# Compile web service input
base64Img = pilImgToBase64(pilImread(imgPath))

data = '{"input_df": "%s"}' % base64Img
print(data)
# body = str.encode(json.dumps(data))
# print(body)
# print(data)


headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ service_key)}

response = requests.post(cluster_scoring_url, data, headers=headers, timeout=60)
if response.status_code != 200:
    print("error code:", response.status_code)
resp = re.sub(r"\\", "", response.content.decode("utf-8")).strip("\"")
resp = json.loads(resp)

print(resp)
all_peaks = np.array(resp["all_peaks"])
candidate = np.array(resp["candidate"])
executionTimeMs = resp["executionTimeMs"]
subset = np.array(resp["subset"])


# print subset
stickwidth = 4

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
cmap = matplotlib.cm.get_cmap('hsv')

imgPath
canvas = cv2.imread(imgPath)
oriImg = cv2.imread(imgPath)

# B,G,R order

for i in range(18):
    rgba = np.array(cmap(1 - i / 18. - 1. / 36))
    rgba[0:3] *= 255
#     print(all_peaks[i])
    for j in range(len(all_peaks[i])):
#         print(all_peaks[i][j][0:2])
        cv2.circle(canvas, tuple(map(int, all_peaks[i][j][0:2])), 4, colors[i], thickness=-1)
# plt.imshow(canvas[:, :, [2, 1, 0]])
# plt.show()
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

to_plot = cv2.addWeighted(oriImg, 0.3, canvas, 0.7, 0)

for i in range(17):
    for n in range(len(subset)):
        index = subset[n][np.array(limbSeq[i]) - 1]
        if -1 in index:
            continue
        cur_canvas = canvas.copy()
        Y = candidate[index.astype(int), 0]
        X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.show()
