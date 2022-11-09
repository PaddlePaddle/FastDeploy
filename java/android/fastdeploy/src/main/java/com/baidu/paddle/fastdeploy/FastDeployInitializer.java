package com.baidu.paddle.fastdeploy;

/**
 * Initializer for FastDeploy. The initialization methods are called by package
 * classes only. Public users don't have to call them. Public users can get
 * FastDeploy information constants such as JNI lib name in this class.
 */
public class FastDeployInitializer {
    /** name of C++ JNI lib */
    public final static String JNI_LIB_NAME = "fastdeploy_jni";

    /**
     * loads the C++ JNI lib. We only call it in our package, so it shouldn't be
     * visible to public users.
     *
     * @return true if initialize successfully.
     */
    public static boolean init() {
        System.loadLibrary(JNI_LIB_NAME);
        return true;
    }
}
