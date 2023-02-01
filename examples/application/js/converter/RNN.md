English | [简体中文](RNN_CN.md)
# The computation process of RNN operator

## 1. Understanding of RNN

**RNN** is a recurrent neural network, including an input layer, a hidden layer and an output layer, which is specialized in processing sequential data.

![RNN](https://user-images.githubusercontent.com/43414102/144739164-d6c4b9ff-d885-4812-8d05-5bf045d3a11b.png)
paddle official document: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/RNN_cn.html#rnn

paddle source code implementation: https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/rnn_op.h#L812

## 2. How to compute RNN 

 At moment t, the input layer is ![图片](https://paddlejs.bj.bcebos.com/doc/xt.svg), hidden layer is ![图片](https://paddlejs.bj.bcebos.com/doc/st.svg), output layer is  ![图片](https://paddlejs.bj.bcebos.com/doc/ot.svg). As the picture above, ![图片](https://paddlejs.bj.bcebos.com/doc/st.svg)isn't just decided by  ![图片](https://paddlejs.bj.bcebos.com/doc/xt.svg),it is also related to ![图片](https://paddlejs.bj.bcebos.com/doc/st1.svg). The formula is as follows.:

![RNN公式](https://user-images.githubusercontent.com/43414102/144739185-92724c8c-25f7-4559-9b1d-f1d76e65d965.jpeg)

## 3.  RNN operator implementation in pdjs

Because the gradient disappearance problem exists in RNN, and more contextual information cannot be obtained, **LSTM (Long Short Term Memory)** is used in CRNN, which is a special kind of RNN that can preserve long-term dependencies.

Based on the image sequence, the two directions of context are mutually useful and complementary. Since the LSTM is unidirectional, two LSTMs, one forward and one backward, are combined into a **bidirectional LSTM**. In addition, multiple layers of bidirectional LSTMs can be stacked. ch_PP-OCRv2_rec_infer recognition model is using a two-layer bidirectional LSTM structure. The calculation process is shown as follows.

#### Take ch_ppocr_mobile_v2.0_rec_infer model, rnn operator as an example
```javascript
{
	Attr: {
		mode: 'LSTM'
		//  Whether bidirectional, if true, it is necessary to traverse both forward and reverse.
		is_bidirec: true
		// Number of hidden layers, representing the number of loops.
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

#### Overall computation process
![LSTM计算过程](https://user-images.githubusercontent.com/43414102/144739246-daf839ad-1d96-4e1d-8f34-38ed0bc5f288.png)
#### Add op in rnn calculation
1) rnn_origin
Formula: blas.MatMul(Input,  WeightList_ih, blas_ih) + blas.MatMul(PreState,  WeightList_hh,  blas_hh)

2) rnn_matmul
Formula: rnn_matmul = rnn_origin +  Matmul( $ S_{t-1} $,  WeightList_hh)

3) rnn_cell
Method: Split the rnn_matmul op output into 4 copies, each copy performs a different activation function calculation, and finally outputs lstm_x_y.tmp_c[1,  1,  48]. x∈[0, 3], y∈[0, 24].
For details, please refer to [rnn_cell](https://github.com/PaddlePaddle/Paddle.js/blob/release/v2.2.5/packages/paddlejs-backend-webgl/src/ops/shader/rnn/rnn_cell.ts).


4) rnn_hidden
Split the rnn_matmul op output into 4 copies, each copy performs a different activation function calculation, and finally outputs lstm_x_y.tmp_h[1,  1,  48]. x∈[0, 3], y∈[0, 24].
For details, please refer to [rnn_hidden](https://github.com/PaddlePaddle/Paddle.js/blob/release/v2.2.5/packages/paddlejs-backend-webgl/src/ops/shader/rnn/rnn_hidden.ts).


