

caffe模型转pytorch



语义分割任务中如何处理label为255的标签
https://blog.csdn.net/qq_43152622/article/details/122663400


np.bincount()用在分割领域生成混淆矩阵
https://blog.csdn.net/weixin_45377629/article/details/124237272



torch.return_types.max(values=tensor([62,  6, 65]),indices=tensor([2, 3, 1]))
torch.max返回2个值


CrossEntropyLoss两种输入



数组转image，resize到固定尺度(0-255像素，batch需要去掉)
Image.fromarray(voc_array)







python demo.py single --config-path configs/voc12.yaml --model-path ~/Downloads/models/deeplabv2_resnet101_msc-vocaug-20000.pth --image-path /Users/admin/Downloads/cartest/cartest0.jpg