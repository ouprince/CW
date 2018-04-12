## 链接
	https://github.com/ouprince/CW.git
	
## 说明
	原版论文：《CW聚类算法.pdf》
	作者翻译：《CW聚类算法论文翻译.doc》
	该代码在 python 2.7 环境运行正常

## 代码运行原理
	(1) 将句子设成 networkx 图形的节点
	(2) 采用 KDRree算法 和 Lucene 搜索节点邻域
	(3) 仅计算节点和其邻域句子的相似度，作为节点边之间的权重，同时设置阈值，相似度如果小于此阈值
		则不加入边。这么做的目的是为了减少算法的时间复杂度。如果有 m 个句子，则原本时间复杂度为 m*m，
		仅计算邻域可以下降为 m*n (n就是取邻域的个数，比如取 10)。同时设定阈值可以降低 Chinese-whispers 
		算法的计算量，不加入多余的边。
	(4) 开始Chinese-whispers 算法，首先将每个节点用它们的id 定义一个类，这样 m 个节点就有 m 个类，类名就是它们的 id。
	(5) 依次计算每个节点与它相邻的最大类，并更新节点类为最大类。一直迭代此过程，直到达到迭代次数。
	(6) 去除类别没变的孤立节点，生成聚类结果。
	
## 代码简介
	(1) common.word2vec 模块使用 numpy 模块将中文的二进制编码转换成 100 维的向量
		加载成一个 22 万词汇左右的一个词向量模型
	(2) common.utils 模块主要是一些编码转换和smart_open 函数
	(3) common.tokenizer 模块主要是分词工具和预处理
	(4) common.similarity 模块用来计算句子的相似度，采用 欧式距离 和 词汇匹配 的加权作为计算结果
	(5) text.search.vtree 模块就是创建 kd-tree 并实现领域查找
	(6) text.search.posting 模块是 lucene 检索模块
	(7) text.clustering.cw 模块就是 Chinese-whispers 算法实现模块
	
## 代码测试
	cd app/text/clustering
	python test.py
