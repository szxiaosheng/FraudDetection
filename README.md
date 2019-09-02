# FraudDetection
在传统贸易企业中，销售人员经常会执行一些欺诈交易以便带来个人收益，实验利用最大似然估计的神经网络方案，用于检测当前交易的欺诈嫌疑，达到95.1%准确率，在此个案上，大大优于贝叶斯神经网络的方案。

# FraudDetection
In traditional trading companies, salespeople often perform some fraudulent transactions in order to bring personal gains. Experiments use the neural network scheme of maximum likelihood estimation to detect fraud suspects in current transactions, achieving 95.1% accuracy. , greatly better than the Bayesian neural network scheme.

# 常见问题
Q : 特征量包括哪些？  
A : 客户编码、物料型号、品牌、送货区域、送货地址、发货数量、发货金额、发货成本、利润额、毛利百分比、销售员、销售员发货时在职月份数、销售员籍贯；  
  
Q : 预测及学习标签是哪一列？  
A ：最后一列表示欺诈标签。  
  
Q ：为什么文件中的的型号、地址、销售员等全是数字？  
A ：在机器学习中，任何文字类型都必须转换成机器可以用于计算的数字形式，否则任何数学公司都没办法计算；所有必须需要存在一个数据预处理的过程。  

Q ：通常的数据处理过程是怎么的？  
A ：通常数据处理过程将包括数据范围定义、数据来源寻找、非结构化转换结构化、特征变量猜想、特征变量数字化；然后才进入模型选择以及数据训练的过程；最后才是把训练模型接口化，供外部应用程序调用。  

Q :该程序目前在MAC机器上运行，如果运行报文件写入错误的否，如何处理？  
A ：首先 sudo touch file.png; 其次 sudo chmod 777 file.png  


