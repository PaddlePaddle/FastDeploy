# RNN算子计算过程

## 一、RNN理解

**RNN** 是循环神经网络，由输入层、隐藏层和输出层组成，擅长对序列数据进行处理。

![RNN](https://user-images.githubusercontent.com/43414102/144739164-d6c4b9ff-d885-4812-8d05-5bf045d3a11b.png)
paddle官网文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/RNN_cn.html#rnn

paddle源码实现：https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/rnn_op.h#L812

##二、RNN计算方式

 t 时刻，输入层为 ![图片](https://paddlejs.bj.bcebos.com/doc/xt.svg) ，隐藏层为 ![图片](https://paddlejs.bj.bcebos.com/doc/st.svg) ，输出层为 ![图片](https://paddlejs.bj.bcebos.com/doc/ot.svg)  。由上图可知，![图片](https://paddlejs.bj.bcebos.com/doc/st.svg) 的值不仅仅取决于 ![图片](https://paddlejs.bj.bcebos.com/doc/xt.svg)  ，还取决于 ![图片](https://paddlejs.bj.bcebos.com/doc/st1.svg) 。计算公式如下：

![RNN公式](https://user-images.githubusercontent.com/43414102/144739185-92724c8c-25f7-4559-9b1d-f1d76e65d965.jpeg)

## 三、pdjs中RNN算子实现

因为 RNN 有梯度消失问题，不能获取更多上下文信息，所以 CRNN 中使用的是 **LSTM（Long Short Term Memory）**，LSTM 是一种特殊的 RNN，能够保存长期的依赖关系。

基于图像的序列，两个方向的上下文是相互有用且互补的。由于 LSTM 是单向的，所以将两个 LSTM，一个向前和一个向后组合到一个**双向 LSTM** 中。此外，可以堆叠多层双向 LSTM。ch_PP-OCRv2_rec_infer 识别模型就是使用的双层双向 LSTM 结构。计算过程如下图所示：

#### 以ch_ppocr_mobile_v2.0_rec_infer 模型 rnn算子为例：
```javascript
{
	Attr: {
		mode: 'LSTM'
		//  是否双向，为true则正向反向都需要遍历
		is_bidirec: true
		// 隐藏层层数，代表循环次数
		num_layers: 2
	}
	
	Input: [
		transpose_1.tmp_0[25, 1, 288]
	]

	PreState: [
		fill_constant_batch_size_like_0.tmp_0[4, 1, 48],  
		fill_constant_batch_size_like_1.tmp_0[4, 1, 48]
	]

	WeightList: [
		lstm_cell_0.w_0[192, 288], lstm_cell_0.w_1[192, 48], 
		lstm_cell_1.w_0[192, 288], lstm_cell_1.w_1[192, 48],
		lstm_cell_2.w_0[192, 96], lstm_cell_2.w_1[192, 48], 
		lstm_cell_3.w_0[192, 96], lstm_cell_3.w_1[192, 48],
		lstm_cell_0.b_0[192], lstm_cell_0.b_1[192],
		lstm_cell_1.b_0[192], lstm_cell_1.b_1[192],
		lstm_cell_2.b_0[192], lstm_cell_2.b_1[192], 
		lstm_cell_3.b_0[192], lstm_cell_3.b_1[192]
	]

	Output: [
	    lstm_0.tmp_0[25, 1, 96]
    ]
}
```

#### 整体计算过程
![LSTM计算过程](https://user-images.githubusercontent.com/43414102/144739246-daf839ad-1d96-4e1d-8f34-38ed0bc5f288.png)
#### rnn 计算中新增op：
1）rnn_origin

计算公式： blas.MatMul(Input,  WeightList_ih, blas_ih) + blas.MatMul(PreState,  WeightList_hh,  blas_hh)

2）rnn_matmul

计算公式：rnn_matmul = rnn_origin +  Matmul( $ S_{t-1} $,  WeightList_hh)

3）rnn_cell

计算方式：将rnn_matmul op输出结果分割成4份，每份执行不同激活函数计算，最后输出lstm_x_y.tmp_c[1,  1,  48]。x∈[0, 3]，y∈[0, 24]。
详见算子实现：[rnn_cell](../paddlejs-backend-webgl/src/ops/shader/rnn/rnn_cell.ts)
)

4）rnn_hidden
计算方式：将rnn_matmul op输出结果分割成4份，每份执行不同激活函数计算，最后输出lstm_x_y.tmp_h[1,  1,  48]。x∈[0, 3]，y∈[0, 24]。
详见算子实现：[rnn_hidden](../paddlejs-backend-webgl/src/ops/shader/rnn/rnn_hidden.ts)


