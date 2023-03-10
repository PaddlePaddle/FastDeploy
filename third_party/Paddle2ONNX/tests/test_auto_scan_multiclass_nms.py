from multiprocessing import Process
from multiprocessing import Queue
import numpy as np


def all_sort(x):
    x1 = x.T
    y = np.split(x1, len(x1))
    z = list(reversed(y))
    index = np.lexsort(z)
    return x[index]


def untar(tar_file, save_path):
    import tarfile
    tf = tarfile.open(tar_file)
    tf.extractall(save_path)


from detection_ops.nms import multiclass_nms


def gen_paddle_nms(q):
    import paddle
    paddle.set_device("cpu")

    class Model(paddle.nn.Layer):
        def __init__(self, score_threshold, nms_top_k, keep_top_k,
                     nms_threshold, normalized, nms_eta, background_label,
                     return_index, return_rois_num):
            super(Model, self).__init__()
            self.score_threshold = score_threshold
            self.nms_top_k = nms_top_k
            self.keep_top_k = keep_top_k
            self.nms_threshold = nms_threshold
            self.normalized = normalized
            self.nms_eta = nms_eta
            self.background_label = background_label
            self.return_index = return_index
            self.return_rois_num = return_rois_num

        def forward(self, bboxes, scores):
            return multiclass_nms(
                bboxes,
                scores,
                score_threshold=self.score_threshold,
                nms_top_k=self.nms_top_k,
                keep_top_k=self.keep_top_k,
                nms_threshold=self.nms_threshold,
                normalized=self.normalized,
                nms_eta=self.nms_eta,
                background_label=self.background_label,
                return_index=self.return_index,
                return_rois_num=self.return_rois_num)

    config = dict()
    score_threshold = np.random.uniform(0.0, 0.8)
    nms_top_k = int(np.random.uniform(10, 200))
    keep_top_k = int(np.random.uniform(10, 200))
    nms_threshold = np.random.uniform(0.0, 0.8)
    normalized = np.random.uniform(0.0, 1.0) > 0.5
    nms_eta = 1.0
    background_label = int(np.random.uniform(0.0, 8.0))
    return_index = np.random.uniform(0.0, 1.0) > 0.5
    return_rois_num = True
    print("===============================\n")
    print({
        "score_threshold": score_threshold,
        "nms_top_k": nms_top_k,
        "keep_top_k": keep_top_k,
        "nms_threshold": nms_threshold,
        "normalized": normalized,
        "nms_eta": nms_eta,
        "background_label": background_label,
        "return_index": return_index,
        "return_rois_num": return_rois_num
    })
    print("\n===============================\n")
    model = Model(
        score_threshold=score_threshold,
        nms_top_k=nms_top_k,
        keep_top_k=keep_top_k,
        nms_threshold=nms_threshold,
        normalized=normalized,
        nms_eta=nms_eta,
        background_label=background_label,
        return_index=return_index,
        return_rois_num=True)
    model.eval()
    ipt0 = paddle.static.InputSpec(
        dtype='float32', shape=[-1, 22743, 4], name='x0')
    ipt1 = paddle.static.InputSpec(
        dtype='float32', shape=[-1, 80, 22743], name='x1')
    paddle.jit.save(model, 'nms/model', [ipt0, ipt1])
    q.put(True)


def gen_onnx_export(q):
    import paddle
    paddle.set_device('cpu')
    import pickle
    untar('detection_ops/nms_inputs.tar.gz', '.')
    data = [np.load('nms_ipt0.npy'), np.load('nms_ipt1.npy')]
    model = paddle.jit.load('nms/model')
    result = model(paddle.to_tensor(data[0]), paddle.to_tensor(data[1]))
    if not isinstance(result, list):
        result = [result]
    result0 = [np.array(r) for r in result]

    for opset in range(10, 16):
        import paddle2onnx
        onnx_file_path = "nms/nms_{}.onnx".format(opset)
        paddle2onnx.export(
            "nms/model.pdmodel",
            "",
            onnx_file_path,
            opset,
            auto_upgrade_opset=False,
            verbose=True,
            enable_onnx_checker=True,
            enable_experimental_op=True)
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_file_path)
        result1 = sess.run(None, {"x0": data[0], "x1": data[1]})
        assert (len(result0) == len(result1),
                "multiclass_nms3: Length of result is not same")
        diff = np.fabs(all_sort(result0[0]) - all_sort(result1[0]))
        print("Max diff of BBoxes:", result0[0].shape, result1[0].shape,
              diff.max())
        assert diff.max(
        ) < 1e-05, "Difference={} of bbox is exceed 1e-05".format(diff.max())
        for i in range(1, len(result0)):
            diff = np.fabs(result0[i] - result1[i])
            print(result0[i], result1[i])
            assert diff.max(
            ) < 1e-05, "Difference={} of output {}(shape is {}) is exceed 1e-05".format(
                diff.max(), i, result0[i].shape)
        q.put(True)


def test_nms():
    for i in range(100):
        q0 = Queue()
        p0 = Process(target=gen_paddle_nms, args=(q0, ))
        p0.start()
        p0.join()
        if not q0.get(timeout=1):
            assert false, "Test failed for multiclass_nms as gen paddle model step."
        q1 = Queue()
        p1 = Process(target=gen_onnx_export, args=(q1, ))
        p1.start()
        p1.join()
        if not q1.get(timeout=1):
            assert false, "Test failed for multiclass_nms at gen_onnx_export step."


if __name__ == "__main__":
    test_nms()
