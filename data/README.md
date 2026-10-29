# data/

`basic_export.csv`（约 20 MB，HaGRID 派生的 837 条手部关键点）**不随仓库提交**，
只有重新训练判指分类器时才需要。

下载方式：

```bash
curl -L -o data/basic_export.csv \
  https://hf-mirror.com/datasets/GestureDetectionConnoisseurs/preprocessed_hand_skeletons/resolve/main/basic_export.csv
```

原始数据集：
<https://huggingface.co/datasets/GestureDetectionConnoisseurs/preprocessed_hand_skeletons>

然后运行：

```bash
python tools/train_finger_model.py --csv data/basic_export.csv
```
