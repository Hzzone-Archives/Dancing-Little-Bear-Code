## prepare the environment
```shell
cd ~
sudo apt update
sudo apt install -y python3 python3-pip python-opencv openmpi-bin python3-tk
git clone https://github.com/Hzzone/Dancing-Little-Bear-Code.git
cd Dancing-Little-Bear-Code
wget https://cntk.ai/PythonWheel/CPU-Only/cntk-2.3.1-cp35-cp35m-linux_x86_64.whl
cd Human-Pose-Estimation
pip3 install azure-ml-api-sdk==0.1.0a10 azureml.datacollector==0.1.0a13 azureml-requirements -i https://azuremldownloads.azureedge.net/python-repository/preview --extra-index-url https://pypi.python.org/simple
```


### 登陆
```shell
az login
```

### 设置默认账户
```shell
az account list -o table
az account set -s 0ca618d2-22a8-413a-96d0-0f1b531129c3     <--- Boston DS Dev, I whitelisted this.
```

### 搭建容器和选择区域，资源组
```shell
az ml env setup -n hzzoneml -l eastus2
```

### 查看搭建情况
```shell
az ml env show -g hzzonemlrg -n hzzoneml
```

### 设置当前环境和资源组
```shell
az ml env set -g hzzonemlrg -n hzzoneml
```


### 创建管理账户
```shell
az ml account modelmanagement create -n hzzoneadmin -g hzzonemlrg -l eastus2
```

```
{
  "created_on": "2018-03-11T12:39:09.060982Z",
  "description": "",
  "id": "/subscriptions/5b797442-b115-43f7-947a-93dca2551e74/resourceGroups/hzzonemlrg/providers/Microsoft.MachineLearningModelManagement/accounts/hzzoneadmin",
  "location": "eastus2",
  "model_management_swagger_location": "https://eastus2.modelmanagement.azureml.net/api/subscriptions/5b797442-b115-43f7-947a-93dca2551e74/resourceGroups/hzzonemlrg/accounts/hzzoneadmin/swagger.json?api-version=2017-09-01-preview",
  "modified_on": "2018-03-11T12:39:09.060982Z",
  "name": "hzzoneadmin",
  "resource_group": "hzzonemlrg",
  "sku": {
    "capacity": 1,
    "name": "S1"
  },
  "subscription": "5b797442-b115-43f7-947a-93dca2551e74",
  "tags": {},
  "type": "Microsoft.MachineLearningModelManagement/accounts"
}
```



### 创建即时服务，上传文件
az ml service create realtime -f deploymain.py -s outputs/schema.json -n imapp3 -v -r python -c aml_config/conda_dependencies.yml -d util.py -d download_model.py --model-file model/pose_net.cntkmodel

```
root@VM-156-20-ubuntu:~/Human-Pose-Estimation# az ml service create realtime -f deploymain.py -s outputs/schema.json -n imapp3 -v -r python -c aml_config/conda_dependencies.yml -d util.py -d download_model.py --model-file model/pose_net.cntkmodel
Starting service create
Starting image create
Starting manifest create
Uploading pipRequirements file
Uploading condaEnvFile file
Starting model register
 model/pose_net.cntkmodel
Attempting to register model to https://eastus2.modelmanagement.azureml.net/api/subscriptions/5b797442-b115-43f7-947a-93dca2551e74/resourceGroups/hzzonemlrg/accounts/hzzoneadmin/models
Attempting to register model with this information: {u'mimeType': u'application/json', u'description': u'', u'tags': [], u'url': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/3e514c88-7602-4057-8416-2ad5a9766f5b.tar.gz?sr=b&sp=r&sig=vdhg0kWG4mt0W6QpizYzCdG6opSDT6emym%2B8Iul2A4U%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A34Z', u'unpack': True, u'name': 'pose_net.cntkmodel'}
Model register post url: https://eastus2.modelmanagement.azureml.net/api/subscriptions/5b797442-b115-43f7-947a-93dca2551e74/resourceGroups/hzzonemlrg/accounts/hzzoneadmin/models
Successfully registered model
Id: cc6a9e840b08415a9e0b3e29c5aeb17c
More information: 'az ml model show -m cc6a9e840b08415a9e0b3e29c5aeb17c'
Creating new driver at /tmp/tmpTDD7Xo.py
Driver uploaded to http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/35bac4ce-ce28-4da2-8e4e-866abf8a6b49.py?sr=b&sp=r&sig=DqV%2B0Kw6NghrqI5WUmQFiRCe0MEdsCQRPMvK%2Bfzdeag%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A35Z
 util.py
Added dependency util.py to assets.
 download_model.py
Added dependency download_model.py to assets.
 deploymain.py
Added dependency deploymain.py to assets.
 outputs/schema.json
Added dependency outputs/schema.json to assets.
Manifest payload: {u'description': None, u'driverProgram': u'driver', u'modelType': u'Registered', u'name': 'imapp3', u'modelIds': [u'cc6a9e840b08415a9e0b3e29c5aeb17c'], u'targetRuntime': {u'properties': {'pipRequirements': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/requirementssH2udB.txt?sr=b&sp=r&sig=XNPvl3vTQXW6noVMwlwJ9Ky8lXg/6xY%2Bh198d5hTHJU%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A02Z', 'condaEnvFile': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/conda_dependencies.yml?sr=b&sp=r&sig=qodpl1let6DwY4w2adSOLmLmZphsSD6jYpn73t64E5Y%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A03Z'}, u'runtimeType': 'Python'}, u'assets': [{'url': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/35bac4ce-ce28-4da2-8e4e-866abf8a6b49.py?sr=b&sp=r&sig=DqV%2B0Kw6NghrqI5WUmQFiRCe0MEdsCQRPMvK%2Bfzdeag%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A35Z', 'mimeType': 'application/x-python', 'id': 'driver'}, {'mimeType': 'application/octet-stream', 'url': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/0c8c9995-8aaa-4be0-b799-9472075c45e5.tar.gz?sr=b&sp=r&sig=GWO7Fd7c9vAqTgQM/srLS0DFiEVOrVWLjMASwN68lSA%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A35Z', 'unpack': True, 'id': '0c8c9995-8aaa-4be0-b799-9472075c'}, {'mimeType': 'application/octet-stream', 'url': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/8c0d37a6-cc23-4cea-ba6d-1b286bfa89db.tar.gz?sr=b&sp=r&sig=LbsBmP/CHhI5ovleIdZyCIYJr%2BrV9M65RGFh6vce07Q%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A36Z', 'unpack': True, 'id': '8c0d37a6-cc23-4cea-ba6d-1b286bfa'}, {'mimeType': 'application/octet-stream', 'url': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/84c0f76a-ffd1-4a01-9cb9-1e1b91c05016.tar.gz?sr=b&sp=r&sig=MXj6kf70xdXwoFirlNsn3EwPJFYNdEC0yiG23MmFVnM%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A36Z', 'unpack': True, 'id': '84c0f76a-ffd1-4a01-9cb9-1e1b91c0'}, {'mimeType': 'application/octet-stream', 'url': 'http://mlcrpstg28d64569d5b9.blob.core.windows.net/amlbdpackages/1b441c09-9845-496f-b7d4-ea23132cd85a.tar.gz?sr=b&sp=r&sig=xPe/5LfVGID/iezIyENk3ell0ZxBQJwcrl9V/CFIKw8%3D&sv=2017-04-17&se=2018-04-10T12%3A40%3A36Z', 'unpack': True, 'id': '1b441c09-9845-496f-b7d4-ea23132c'}]}
Successfully created manifest
Id: 09441506-3efa-43f4-b135-a00f645e0171
More information: 'az ml manifest show -i 09441506-3efa-43f4-b135-a00f645e0171'
Image payload: {u'computeResourceId': '/subscriptions/5b797442-b115-43f7-947a-93dca2551e74/resourcegroups/hzzonemlrg/providers/Microsoft.MachineLearningCompute/operationalizationClusters/hzzoneml', u'manifestId': u'09441506-3efa-43f4-b135-a00f645e0171', u'name': 'imapp3', u'imageType': u'Docker', u'description': ''}
Operation Id: 23ab39ed-0bb4-4389-a131-c321fbf9ee5d
Creating image.......................................................................Done.
Image ID: 0e99589b-3a73-416e-8216-2289049972da
More details: 'az ml image show -i 0e99589b-3a73-416e-8216-2289049972da'
Usage information: 'az ml image usage -i 0e99589b-3a73-416e-8216-2289049972da'
[Local mode] Running docker container.
[Local mode] Pulling the image from mlcrpacr0984963767ad.azurecr.io/imapp3:1. This may take a few minutes, depending on your connection speed...
[Local mode] Pulling.......................................................................
Container port: 32791
[Local mode] Waiting for container to initialize.[Debug] Fetching sample data from swagger endpoint: http://127.0.0.1:32791/swagger.json

[Local mode] Done
[Local mode] Service ID: imapp3
[Local mode] Usage: az ml service run realtime -i imapp3 -d "{u\"input_df\": u\"sample data text\"}"
[Local mode] Additional usage information: 'az ml service usage realtime -i imapp3'
```


### 测试
```shell
az ml service run realtime -i imapp3 -d "{\"input_df\": \"data:image/jpeg;base64,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\"}"
```

### 查看输出
```shell
az ml service logs realtime -i imapp3
```


### 查看当前如何使用
```shell
az ml service usage realtime -i hzzonemlservice
```

```
Scoring URL:
    http://127.0.0.1:32797/score

Headers:
    Content-Type: application/json

Swagger URL:
    http://127.0.0.1:32797/swagger.json

Sample CLI command:
    az ml service run realtime -i hzzonemlservice -d "{u\"input_df\": u\"sample data text\"}"

Sample CURL call:
    curl -X POST -H "Content-Type:application/json" --data "{u\"input_df\": u\"sample data text\"}" http://127.0.0.1:32797/score

Get debug logs by calling:
    az ml service logs realtime -i hzzonemlservice
```





