import paddle.fluid as fluid
import sys
import paddle
paddle.enable_static()

model_dir = sys.argv[1]
new_model_dir = sys.argv[2]
exe = fluid.Executor(fluid.CPUPlace())
[inference_program, feed_target_names,
 fetch_targets] = fluid.io.load_inference_model(
     dirname=model_dir, executor=exe)

print(feed_target_names)
fluid.io.save_inference_model(
    dirname=new_model_dir,
    feeded_var_names=feed_target_names,
    target_vars=fetch_targets,
    executor=exe,
    main_program=inference_program,
    params_filename="__params__")
