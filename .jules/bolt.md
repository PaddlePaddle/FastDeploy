## 2024-05-22 - Performance Analysis of RowParallelLinear

**Learning:** `RowParallelLinear` has `all2all_transpose`.
```python
    def all2all_transpose(self, x: paddle.Tensor) -> paddle.Tensor:
        token_num = x.shape[0]
        token_num_pad = (token_num + self.tp_size - 1) // self.tp_size * self.tp_size
        if self.fd_config.scheduler_config.splitwise_role == "decode" and not current_platform.is_xpu():
            if not (token_num_pad > token_num):
                x_padded = x
            else:
                x_padded = paddle.zeros([token_num_pad, x.shape[1]], x.dtype)
                x_padded[:token_num] = x
            out = paddle.zeros([token_num_pad // self.tp_size, x.shape[1] * self.tp_size], x.dtype)
            decode_alltoall_transpose(x_padded, out)
        else:
            if token_num_pad > token_num:
                x_new = paddle.zeros([token_num_pad, x.shape[1]], x.dtype)
                x_new[:token_num, :] = x
                x = x_new
            out = paddle.zeros_like(x)
            paddle.distributed.alltoall(out, x, group=self.tp_group)
            out.reshape_([self.tp_size, -1, x.shape[1]])
            out = paddle.transpose(out, [1, 0, 2])
            out.reshape_([x.shape[0] // self.tp_size, self.input_size])
        return out
```

Here `x_padded` and `out` are initialized with `paddle.zeros`.
`x_padded` is populated with `x` and then the rest is padding. If `decode_alltoall_transpose` expects zeros in padding, then `zeros` is needed.
However, `out` is the output buffer. It is definitely overwritten by `decode_alltoall_transpose` or `paddle.distributed.alltoall`.
So `out` can be `paddle.empty`.

Also `x_new` (in the `else` block) handles padding.
`out = paddle.zeros_like(x)` in the `else` block is the receive buffer for `alltoall`. It is overwritten. It should be `paddle.empty_like`.

**Plan:**
1.  Optimize `allgather` in `fastdeploy/model_executor/layers/normalization.py`. Use `paddle.empty`.
2.  Optimize `all2all_transpose` in `fastdeploy/model_executor/layers/linear.py`. Use `paddle.empty` for output buffers.

**Impact:**
Allocating zero-filled memory is slower than just allocating memory, especially for large tensors involved in distributed communication. This is a recurring pattern in the communication layers.

Let's verify `decode_alltoall_transpose` semantics. I don't see the code for it (it is imported), but standard communication ops overwrite the output buffer.

Let's start with `fastdeploy/model_executor/layers/normalization.py`.
The `allgather` method:
```python
    def allgather(self, out, token_num):
        """
        Gather the output tensor from each tensor parallel rank.

        Args:
            out (paddle.Tensor): Output tensor to be gathered.

        Returns:
            paddle.Tensor: Gathered tensor.
        """
        token_num_per_rank = out.shape[0]
        if token_num_per_rank == 0:
            return out
        multi_outs = paddle.zeros([token_num_per_rank * self.tp_size, out.shape[1]], dtype=out.dtype)
        paddle.distributed.all_gather(multi_outs, out, self.tp_group)
        return multi_outs[:token_num, :]
```

Replacing `paddle.zeros` with `paddle.empty` here is safe because `all_gather` fills the entire `multi_outs` tensor (since `multi_outs` size is exactly `token_num_per_rank * tp_size` and each rank sends `token_num_per_rank` rows).

Wait, is `token_num_per_rank` uniform across ranks?
In `split`:
```python
        token_num = x.shape[0]
        token_num_per_rank = (token_num + self.tp_size - 1) // self.tp_size
```
This is ceiling division.
If `token_num` = 10, `tp_size` = 4. `token_num_per_rank` = 3.
Rank 0: 0-3 (3 items)
Rank 1: 3-6 (3 items)
Rank 2: 6-9 (3 items)
Rank 3: 9-10 (1 item) -> but `part_x` is size 3. The last 2 items are padding (zeros).

So `out` coming into `allgather` has size 3.
Rank 3 sends 3 items (1 valid, 2 padding).
`all_gather` gathers 4 * 3 = 12 items.
`multi_outs` has size 12.
We return `multi_outs[:10]`.
The padding gathered from Rank 3 is at the end, but we slice it off?
Indices:
R0: 0,1,2
R1: 3,4,5
R2: 6,7,8
R3: 9,10,11
We return 0-9.
Indices 10, 11 are from R3 (padding).
Index 9 is valid from R3.
So the padding is indeed sliced off.

The question is: does `all_gather` require `multi_outs` to be zeroed? No, it overwrites.
Is `out` (the input to `allgather`) guaranteed to have zero padding?
In `split`, `part_x` is initialized with `zeros`.
```python
        part_x = paddle.zeros(shape=[token_num_per_rank, x.shape[1]], dtype=x.dtype)
        part_x[: (end_offset - start_offset), :] = x[start_offset:end_offset, :]
```
If we use `empty` in `split`, `part_x` will have garbage in padding.
Then `out` (result of processing `part_x`) will have garbage in padding (result of garbage input).
Then `all_gather` gathers this garbage.
Then we slice `multi_outs[:token_num]`.
The garbage from `split` (padding) ends up at indices > `token_num`.
So `multi_outs[:token_num]` should be clean.

HOWEVER, `part_x` is processed by the layer (e.g. `RMSNorm` or `Linear`).
In `RMSNorm.forward`:
```python
        if self.split_x:
            assert residual_out is not None
            residual_out = self.split(residual_out)
        if self.allgather_out:
            assert forward_meta is not None
            out = self.allgather(out, forward_meta.ids_remove_padding.shape[0])
```
`split` happens at the end of forward? No, `split` is called on `residual_out`.
Wait, `RMSNorm` seems to support splitting the output for the NEXT layer (which expects split input?).
And `allgather` supports gathering the output (presumably if the input was split?).

If `split` is used, `residual_out` (which is `x` + `residual`) is split.
The padding in `part_x` is garbage if we use `empty`.
This `part_x` is returned as `residual_out`.
The caller of `RMSNorm` will use `residual_out`.
If the caller is a Linear layer in TP, it might rely on the shape but mask the padding?
Usually in TP, we don't rely on padding being zero unless there's a reduction that sums it up.
But `split` partitions the batch/sequence dimension.
If the next layer is `RowParallelLinear`, it takes `x`.
`RowParallelLinear` might use `all2all_transpose` if `split_token` is true.

If I change `split` to `empty`, I risk propagating garbage.
But `allgather` receives `out`. `out` is the result of computation.
In `RMSNorm`, `out` is `norm_out`.
If `allgather_out` is True, `out` is gathered.
`out` here is the LOCAL chunk (if we are in sequence parallel mode).
Wait, if `allgather_out` is True, it implies `out` is currently split/sharded?
Yes.
So `out` has `token_num_per_rank` rows.
And we gather it to form the full tensor.
The padding in `out` (if any) corresponds to the padding we introduced during `split` (or however the input was sharded).
Since we slice `multi_outs[:token_num]`, we discard the padding.
So the value of the padding in `multi_outs` doesn't matter.
So `multi_outs` in `allgather` can definitely be `empty`.

What about `split`?
If `part_x` has garbage padding, and we return it.
This `part_x` goes to the next layer (e.g. MLP).
MLP (Linear -> Act -> Linear).
Linear will multiply garbage. Result is garbage.
Activation (e.g. Swiglu) will operate on garbage. Result is garbage.
Second Linear multiplies garbage. Result is garbage.
Then we `allgather` (or `all_reduce`?).
If we `allgather` at the end of MLP (Sequence Parallel), we discard the garbage padding.
So in theory, garbage padding is fine as long as we don't reduce (sum) it into valid data.
Sequence Parallel usually doesn't reduce across the sequence dim; it gathers.
So `split` using `empty` might also be safe, BUT it's riskier because valid operations (like `norm` or `exp` in Softmax, though Softmax is usually on head dim) might NaN on garbage (e.g. `exp(huge_number)`).
So `split` using `zeros` is safer to avoid NaNs propagating or exceptions.

So I will focus on `allgather` in `RMSNorm` and `all2all_transpose` in `RowParallelLinear` output buffers.

**Action Plan:**
1.  Verify `paddle` availability and `empty` method. (It exists).
2.  Modify `fastdeploy/model_executor/layers/normalization.py`: `allgather` uses `paddle.empty`.
3.  Modify `fastdeploy/model_executor/layers/linear.py`: `all2all_transpose` uses `paddle.empty` for `out`.

Let's double check `all2all_transpose` in `RowParallelLinear`.
```python
            if token_num_pad > token_num:
                x_new = paddle.zeros([token_num_pad, x.shape[1]], x.dtype)
                x_new[:token_num, :] = x
                x = x_new
            out = paddle.zeros_like(x)
```
Here `x` is the input to `alltoall`. `x` might have been padded.
`out` is the output of `alltoall`.
`out` will be overwritten. So `paddle.empty_like(x)` is fine.
Also `decode_alltoall_transpose(x_padded, out)` case. `out` is `paddle.zeros(...)`. `empty` is fine.

One detail: `paddle.empty` might contain NaNs.
If `alltoall` only partially fills `out` (unlikely for `alltoall` which is a full exchange), we might have issues.
`alltoall` exchanges chunks.
Rank i sends chunk j to Rank j.
Rank i receives chunk i from Rank j.
If `token_num_pad` is used, the tensor is perfectly divisible by `tp_size`.
So every chunk is same size.
Every rank sends/receives full chunks.
So `out` is fully written.

So `empty` is safe.

Let's prepare the plan.
I will also check if `paddle.empty` is available in the environment (it should be).

I should also check if `fastdeploy/model_executor/layers/utils.py` has any helpers I should be aware of.
