# images_classification_basline
K-ford
1. 数据存放目录：

    训练数据：
    
    data/train/d1
    
    date/train/d2
    
    ...
    
    测试数据：
    
    test/test
2. 数据增广

    python imgs_aug.py
3. k-ford训练

    nohup python k-ford.py > train.log 2>&1 &
4. 每个ford训练之后的结果进行融合(按照均值)

    python k-ford.py
    
该脚本是k-ford训练和预测脚本
