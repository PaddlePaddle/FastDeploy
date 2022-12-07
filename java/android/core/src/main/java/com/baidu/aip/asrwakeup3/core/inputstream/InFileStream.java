package com.baidu.aip.asrwakeup3.core.inputstream;

import android.app.Activity;
import android.content.Context;

import com.baidu.aip.asrwakeup3.core.util.MyLogger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by fujiayi on 2017/6/20.
 */

public class InFileStream {

    private static Context context;

    private static final String TAG = "InFileStream";

    private static volatile String filename;

    private static volatile InputStream is;

    // 以下3个setContext

    /**
     * 必须要先调用这个方法
     * 如之后调用create16kStream，使用默认的app/src/main/assets/outfile.pcm作为输入
     * 如之后调用createMyPipedInputStream， 见 InPipedStream
     *
     * @param context
     */
    public static void setContext(Context context) {
        InFileStream.context = context;
    }

    /**
     * 使用pcm文件作为输入
     *
     * @param context
     * @param filename
     */
    public static void setContext(Context context, String filename) {
        InFileStream.context = context;
        InFileStream.filename = filename;
    }

    public static void setContext(Context context, InputStream is) {
        InFileStream.context = context;
        InFileStream.is = is;
    }

    public static Context getContext() {
        return context;
    }

    public static void reset() {
        filename = null;
        is = null;
    }


    public static InputStream createMyPipedInputStream() {
        return InPipedStream.createAndStart(context);
    }

    /**
     * 默认使用必须要先调用setContext
     * 默认从createFileStream中读取InputStream
     *
     * @return
     */
    public static InputStream create16kStream() {
        if (is == null && filename == null) {
            // 没有任何设置的话，从createFileStream中读取
            return new FileAudioInputStream(createFileStream());
        }

        if (is != null) { // 默认为null，setInputStream调用后走这个逻辑
            return new FileAudioInputStream(is);
        } else if (filename != null) { //  默认为null， setFileName调用后走这个逻辑
            try {
                return new FileAudioInputStream(filename);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

        return null;
    }

    private static InputStream createFileStream() {
        try {
            // 这里抛异常表示没有调用 setContext方法
            InputStream is = context.getAssets().open("outfile.pcm");
            MyLogger.info(TAG, "create input stream ok " + is.available());
            return is;
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}