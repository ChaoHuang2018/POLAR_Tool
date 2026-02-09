- Install dependencies through apt-get install
```
# 安装依赖（仅导出 JSON 不需要 onnx/onnxruntime）
pip install onnx numpy

# 转 JSON
python txt2net.py --input path/to/model.txt --to json --output model.json

# 转 ONNX
python txt2net.py --input path/to/model.txt --to onnx --output model.onnx

# 同时导两种
python txt2net.py --input model.txt --to json onnx --output model.json model.onnx

# 运行并做一次数值校验（随机采样 5 个点，比对 txt 前向与 JSON 前向是否一致）
python txt2net.py --input model.txt --to json --output model.json --check 5
```
