public class InferDemo {

    private native void infer(String modelPath, String imagePath);

    private final static String JNI_LIB_NAME = "/home/hjyp/code/DeepLearning/FastDeploy/examples/application/java/yolov5/cpp/build/libinferDemo.so";

    static {
        System.load(JNI_LIB_NAME);
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Please input enough params. e.g. java test.java param-dir image-path");
            return;
        }
        String modelPath = args[0];
        String imagePath = args[1];

        InferDemo inferDemo = new InferDemo();

        // 执行推理逻辑
        inferDemo.infer(modelPath, imagePath);
    }
}

