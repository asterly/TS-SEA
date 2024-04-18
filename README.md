# 参考案列
##  Contrastive Clustering  （aaai 2021）
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711940234694-6c858787-67c6-4601-beb2-dd3bb39627ff.png#averageHue=%23f5f2ef&clientId=ue498d113-4a0e-4&from=paste&height=410&id=u09338e7c&originHeight=615&originWidth=1246&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=357150&status=done&style=none&taskId=u8b25cbaf-a7f1-4e5d-88b5-f4c242d9721&title=&width=830.6666666666666)

## Strongly augmented contrastive clustering
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711940662637-9dd8f78a-9b5f-42f8-99cd-cc76ddc37f6e.png#averageHue=%23f2eeea&clientId=ue498d113-4a0e-4&from=paste&height=188&id=u23a349dd&originHeight=282&originWidth=1003&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=67283&status=done&style=none&taskId=u8acbc41e-a488-4f81-aed1-153d596009a&title=&width=668.6666666666666)

![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711939806619-c4c28df9-e08a-425b-90ed-6f3b9ab3c752.png#averageHue=%23f7f5f4&clientId=ue498d113-4a0e-4&from=paste&height=535&id=u9c2776db&originHeight=802&originWidth=1489&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=401645&status=done&style=none&taskId=u72297c66-a896-43c2-af1c-ff838e09f2b&title=&width=992.6666666666666)
##  TEMPO: PROMPT-BASED GENERATIVE PRE-TRAINED TRANSFORMER FOR TIME SERIES FORECASTING   
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711958172083-3c4444c5-e32a-4751-b99f-3d23f0e80567.png#averageHue=%23f2f2f1&clientId=ue498d113-4a0e-4&from=paste&height=565&id=r88t0&originHeight=847&originWidth=1696&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=187969&status=done&style=none&taskId=u55f7f2ce-e378-43cb-ab37-d546da1f5fa&title=&width=1130.6666666666667)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1712036052761-7d842612-1318-421c-91ab-01fcc5ab4cb7.png#averageHue=%23f9f9f9&clientId=uc3cfdbb6-e57a-4&from=paste&height=583&id=u681afc7b&originHeight=874&originWidth=1164&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=167201&status=done&style=none&taskId=u4819461d-17cb-4a25-bbae-c0ebbe056a9&title=&width=776)
# 基线模型
##  ACo-training Approach for Noisy Time Series Learning （cikm 2023 ccf-b） 
在这项工作中，我们专注于稳健的时间序列表示学习。我们的假设是，真实世界的时间序列是嘈杂的，并且同一时间序列的不同视图之间的互补信息在分析嘈杂的输入时起着重要作用。基于这一假设，我们通过两个不同的编码器为输入时间序列创建两个视图。我们通过基于共训练的对比学习迭代地学习这些编码器。我们的实验证明，这种联合训练方法显著提高了性能。特别是，通过利用来自不同视图的互补信息，我们提出的TS-CoT方法可以缓解数据噪声和损坏的影响。在无监督和半监督设置下对四个时间序列基准进行的实证评估表明，TS-CoT优于现有方法。此外，通过微调，TS-CoT学到的表示可以很好地迁移到下游任务中。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711940808352-263c55cc-425e-4c4a-9318-9120137d1bf1.png#averageHue=%23f2f1f1&clientId=ue498d113-4a0e-4&from=paste&height=808&id=uf32c1cc9&originHeight=1212&originWidth=1995&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=380972&status=done&style=none&taskId=u40c3efcb-de54-473f-a0ef-f0aa7901226&title=&width=1330)

### 单视角聚类
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966638221-45f330ab-5f84-4520-a528-3113df4f0afd.png#averageHue=%23f3f3f3&clientId=ube8a3550-03d1-4&from=paste&height=109&id=ua81bf617&originHeight=163&originWidth=892&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=21247&status=done&style=none&taskId=u66eb9960-df64-41c1-b7e4-536d816b48a&title=&width=594.6666666666666)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966647770-b55e7521-0b5c-43f5-a0be-1102aacff82a.png#averageHue=%23efefef&clientId=ube8a3550-03d1-4&from=paste&height=85&id=u05770c04&originHeight=127&originWidth=853&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=20421&status=done&style=none&taskId=ub7070da2-60e9-4f49-b208-02ddc471ec0&title=&width=568.6666666666666)
### 无监督标识学习
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966722559-8c7ace43-1e6f-4298-9170-c09b753a308b.png#averageHue=%23f8f8f8&clientId=ube8a3550-03d1-4&from=paste&height=94&id=u992682a5&originHeight=141&originWidth=784&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=12242&status=done&style=none&taskId=u41a11561-d940-4bed-9a2a-48e5d60a373&title=&width=522.6666666666666)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966731623-ed6b22dd-f47d-4c62-b306-868fdb3bf85e.png#averageHue=%23f9f9f9&clientId=ube8a3550-03d1-4&from=paste&height=89&id=ua4484a04&originHeight=133&originWidth=784&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=10436&status=done&style=none&taskId=uda2e13b9-161a-44f1-a71a-62957d4b883&title=&width=522.6666666666666)

![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966702014-b26b8130-c451-46d9-8d7e-174036fc5e5c.png#averageHue=%23f7f7f7&clientId=ube8a3550-03d1-4&from=paste&height=116&id=u3289ab5e&originHeight=174&originWidth=706&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=15072&status=done&style=none&taskId=u30e13b59-7f66-4268-a008-ec907cd44ed&title=&width=470.6666666666667)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966709553-e56dc6cc-63a1-43bf-a53c-6a97f67484f8.png#averageHue=%23f7f7f7&clientId=ube8a3550-03d1-4&from=paste&height=116&id=udff47a44&originHeight=174&originWidth=703&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=15300&status=done&style=none&taskId=u8b7aa684-d757-490d-a01d-6358ae37cfa&title=&width=468.6666666666667)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966748256-51ed6a19-98d5-467a-b93c-8e5e96c11803.png#averageHue=%23f4f4f4&clientId=ube8a3550-03d1-4&from=paste&height=119&id=ua0fec1e6&originHeight=178&originWidth=799&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=22082&status=done&style=none&taskId=uc8d4450e-ab25-4c88-a763-a4a880689db&title=&width=532.6666666666666)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966758088-abdcde52-0ca7-4fab-bf60-c30065837fc6.png#averageHue=%23f4f4f4&clientId=ube8a3550-03d1-4&from=paste&height=117&id=u5ea3a6a5&originHeight=175&originWidth=805&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=22336&status=done&style=none&taskId=ue9933125-cfd7-4ba6-b4da-3a06c23155b&title=&width=536.6666666666666)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966793106-b4049a59-62dd-4636-8e76-5290ef0ea902.png#averageHue=%23f7f7f7&clientId=ube8a3550-03d1-4&from=paste&height=126&id=u82aac864&originHeight=189&originWidth=670&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=15922&status=done&style=none&taskId=ua521f311-6157-4452-a813-3522491c67c&title=&width=446.6666666666667)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966802353-ac9dd7f5-5248-4ddd-a355-07bdd2de2066.png#averageHue=%23f7f7f7&clientId=ube8a3550-03d1-4&from=paste&height=154&id=ua9f2532c&originHeight=231&originWidth=820&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=20399&status=done&style=none&taskId=ub2ead33e-f939-409a-bb04-d8df1891e20&title=&width=546.6666666666666)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966811739-d7acb263-f36c-4e07-87f0-f6d00a61adb1.png#averageHue=%23f7f7f7&clientId=ube8a3550-03d1-4&from=paste&height=95&id=ue358997d&originHeight=142&originWidth=771&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=13222&status=done&style=none&taskId=u526a83ea-5a56-4ce0-b869-5957216bd82&title=&width=514)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/27646877/1711966819494-b9fccbf7-1d2a-4dbd-b3d6-a247375e8ec0.png#averageHue=%23f6f6f6&clientId=ube8a3550-03d1-4&from=paste&height=75&id=u0f4feaee&originHeight=112&originWidth=768&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=10652&status=done&style=none&taskId=ue6281211-2c68-4b1e-a325-b3bf584044d&title=&width=512)

# 主要结果
### 公开数据数据集
#### 数据描述
|  | HAR | ISRUC | Epilepsy |
| --- | --- | --- | --- |
| Train |  7352  | 6871 |  9200   |
| Test |  2947  | 1718 |  2300   |
| Length |  128  | 3000 |  178   |
| sensors |  9  | 10 | 1 |
| class | 6 | 5 | 2 |

#### 结果

|  | HAR |  |  | ISRUC |  |  | Epilepsy |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  Accuracy | AUROC | F1 |  Accuracy | AUROC | F1 |  Accuracy | AUROC | F1 |
| TS-SEA | 0.94935786 | 0.99743932 | 0.94887217 | 0.82898719 | 0.96842228 | 0.81795838 | 0.98104348 | 0.99593683 | 0.9702041 |
| TS-CoT | 0.93776722 | 0.99599943 | 0.93658227 | 0.81266822 | 0.96366466 | 0.80003021 | 0.97713043 | 0.99597082 | 0.96402357 |
| TS-TCC | 0.91156702 | 0.99040809 | 0.91222396 | 0.80908033 | 0.95993153 | 0.79742465 | 0.97417391 | 0.94672585 | 0.95886258 |
| TS2VEC | 0.939871055 | 0.992266734 | 0.940693532 | 0.772060536 | 0.94149224 | 0.746274914 | 0.972521739 | 0.984829632 | 0.964656052 |


### 桥梁路基段(土木)
#### 数据描述
|  | Bridge | RoadBank |
| --- | --- | --- |
| Train | 849 | 573 |
| Test | 213 | 180 |
| Length | 1500 | 1500 |
| sensors | 16 | 5 |
| class | 2 | 2 |

#### 结果

|  | Bridge |  |  | RoadBank |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  Accuracy | AUROC | F1 |  Accuracy | AUROC | F1 |
| TS-SEA | 0.967778 | 0.966786 | 0.959311 | 0.998432 | 0.997665 | 0.963581 |
| TS-CoT | 0.947222 | 0.96501 | 0.934189 | 0.995483 | 0.99646 | 0.93026 |
| TS-TCC | 0.959259 | 0.942203 | 0.947876 | 0.997491 | 0.928571 | 0.960901 |


# 消融实验
T 表示时域
F 表示频域
S 表示时序分解的季节项
Trn 表示时序分解的趋势项
Res 表示时序分解的残差项

|  |  | HAR |  | ISRUC |  | Epilepsy |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | ACC | F1 | ACC | F1 | ACC | F1 |
| T+S |  | 0.91503224 | 0.91616271 | 0.77532 | 0.758772 | 0.97713 | 0.963785 |
| T+Trn |  | 0.90668476 | 0.90716964 | 0.789988 | 0.771335 | 0.975913 | 0.961755 |
| T+Res |  | 0.91034951 | 0.91021331 | 0.804075 | 0.78686 | 0.974261 | 0.95933 |
| F+S |  | 0.90023753 | 0.89869009 | 0.737718 | 0.713957 | 0.973913 | 0.958779 |
| F+Trn |  | 0.91204615 | 0.90993372 | 0.781607 | 0.753682 | 0.972174 | 0.956145 |
| F+Res |  | 0.90288429 | 0.90061082 | 0.800233 | 0.78108 | 0.975 | 0.960646 |
| S+Trn+Res |  | 0.92480489 | 0.92539703 | 0.792782 | 0.768813 | 0.979043 | 0.967013 |
| T+F+Trn |  | 0.93851374 | 0.93736748 | 0.809197 | 0.789428 | 0.979739 | 0.968222 |
| T+F+Res |  | 0.94536817 | 0.94436286 | 0.826251 | 0.811328 | 0.982522 | 0.972524 |



给定一个时间序列数据集D={X1，X2,...，XN} ，其中Xi= T*dim,T是时间序列样本的长度，dim是每个时间序列的维度或传感器数，y标识时间样本的所属标签。本文的目标是学习三个视角的表示函数g,f和s，分别表示时域、频域，季节项视角。通过多视角的联合学习方案学习不同视角间的特征表示一致性


我们提出的三种视角对比学习的模型整体结构如图所示。我们对原始的时序数据做快速傅里叶变换得到频域视角数据，同时对数据进行趋势项季节项和残差项分解获取时序季节项得到季节性视角数据。

# 致谢
## TS-COT
paper: [A Co-training Approach for Noisy Time Series Learning](https://arxiv.org/abs/2308.12551)

github: [TS-COT](https://github.com/Vicky-51/TS-CoT.git)