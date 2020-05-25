# 
去掉官方的dataloader_utils 放到input_utils.py里面
params_dict移到xl_tensorflow.utils里面

## 问题记录
#### Labels definition in matches.match_results:
        # (1) match_results[i]>=0, meaning that column i is matched with row
        #     match_results[i].
        # (2) match_results[i]=-1, meaning that column i is not matched.
        # (3) match_results[i]=-2, meaning that colum````n i is ignored.
#### 标签类别数量问题
1、与automl不一致（官方版本没有-1,因此类别会有问题），见automl的anchor的552行
2、_unmatched_threshold如果与matched_threshold一致，则不会有-2的情况，见argma_matcher 139行
### 数据加载
 数据加载只返回类别序列，不会返回onehot格式
 
 ### 损失函数
 focal loss 保持一致