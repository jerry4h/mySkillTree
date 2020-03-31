- [ ] Arcface
    - [ ] 手撕 Arcface？[链接](https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py): 拆 Cosine(a+m), one_hot label, one_hot 用来分解 target label 和 others。
    - [x] 简单地解释 Arcface？ArcFace的目的是**增加判别性（减小类内距，同时增大内间距），减弱欠拟合**。是通过设置特征与类内中心点的**margin**来实现的，这个margin约束在角度上，所以叫Arcface，与其他基于margin的工作比较，Arcface有较**清晰的物理模型解释**。
    - [x] Arcface 与 Cosface的区别？直观地看**margin的位置**不同，cosface的margin约束在余弦距离上。核心不同在于，logit-prob曲线上表现优势在于，对**难度样本(小prob)的约束**更强。
    - [x] Arcface 的默认超参数有哪些？scale 和 m。
    - [x] 为什么要 scale？**余弦距离度量范围**在[-1, 1]，如果不scale，交叉熵无法收敛
    - [x] 为什么要 Normalize？以logit为指标时，不同类别受bias影响强烈，特征幅值L2范数较小时，容易受到干扰，故从传统logit舍弃掉了bias，也控制特征幅值L2范数。
    - [x] 为什么特征幅值L2范数较小，易受干扰？因为相同抖动噪声情况下，大L2范数的幅值稳定性更强。
    - [x] Arcface 与 [Focal Loss](../3.3%20检测.md) 的矛盾所在：Arcface 欠拟合，Focal Loss 简单样本过拟合。
    - [ ] 人脸识别的评价指标
- [ ] Triplet loss 的 pytorch [代码](https://discuss.pytorch.org/t/triplet-loss-in-pytorch/30634)：`loss=torch.mean(relu(dist(anchor,pos)+m-dist(anchor,neg)))` 
- [ ] 当前人脸识别损失函数的发展趋势
- [ ] pytorch 实现过哪些层：自定义的损失函数，然后还有 GLFF 论文中也实现过许多融合的策略的可能性，不同的网络结构。
- [ ] 轻量级网络 MobileNet 的各种改进策略？
- [ ] 人脸的未来趋势（VALSE)：Face-based human understanding
    - [ ] 远程生物信号感知（人脸心率估计）
    - [ ] **微表情识别**（下意识的表情，时间短，表情细微，人的判别效果都很差，与人脸取证很像）：**微信号放大**、时域扩增。洪晓鹏（西安交通大学）
    - [x] 人脸其他任务：关注度检测（网课）、疲劳驾驶检测（多子任务pose/gaze/action等）、医疗辅助系统（血氧饱和度等）、人脸防伪、人脸美化
    - [x] 小样本问题：无监督数据扩增、借助图形学等物理模型生成（GAN不可控原因）、多模态想办法多利用样本、模型-数据迭代
    - [ ] 样本不均衡问题：
    - [x] 人脸检测后序子任务问题：质量评估 + 筛选/针对后续问题强化

## Embedding
从高维空间向低维空间的映射

## 行为识别

- [ ] A comprehensive survey of vision-based human action recognition methods
