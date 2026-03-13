# 准备工作
### 1.0 下载测试用的模型
```
bash scripts/download_models.sh
```

### 2.0 修改${PROJECT_ROOT}/configs/hotwords.yaml
浏览网页，修改热词，生成.fst文件，放在${PROJECT_ROOT}/assets/目录下
- https://colab.research.google.com/drive/1jEaS3s8FbRJIcVQJv2EQx19EM_mnuARi?usp=sharing


### 3.0 单个案例测试模型
```
bash scripts/example_sherpa_infer.sh
```