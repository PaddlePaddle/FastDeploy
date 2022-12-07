package com.baidu.aip.asrwakeup3.core.inputstream;

import android.content.Context;

import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;

/**
 * 本示例从app/src/main/assets/outfile.pcm作为byte[]的输入
 * 生成PipedInputStream作为SDK里IN_FILE的参数
 */
public class InPipedStream {

    private PipedInputStream pipedInputStream;
    private PipedOutputStream pipedOutputStream;
    private Context context;

    private InPipedStream(Context context) {
        pipedInputStream = new PipedInputStream();
        pipedOutputStream = new PipedOutputStream();
        this.context = context;
    }

    private void start() throws IOException {
        /**  准备绑定 **/
        pipedInputStream.connect(pipedOutputStream);

        /** 准备文件 **/

        /** 新线程中放入 20ms 音频数据，注意从新线程放入**/
        Runnable run = new Runnable() {
            @Override
            public void run() {

                try {
                    final InputStream is = context.getAssets().open("outfile.pcm");
                    /** 读取20ms的音频二进制数据 放入buffer 中**/
                    int bytePerMs = 16000 * 2 / 1000;
                    int count = bytePerMs * 20; // 20ms 音频数据
                    int r = 0;
                    byte[] buffer = new byte[count];
                    do {
                        r = is.read(buffer);
                        int sleepTime = 0;
                        if (r > 0) {
                            pipedOutputStream.write(buffer, 0, count);
                            sleepTime = r / bytePerMs;
                        } else if (r == 0) {
                            sleepTime = 100; // 这里数值按照自己情况而定
                        }
                        if (sleepTime > 0) {
                            try {
                                Thread.sleep(sleepTime);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                        }

                    } while (r >= 0);
                    is.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        };
        (new Thread(run)).start();
    }

    public static PipedInputStream createAndStart(Context context) {
        InPipedStream obj = new InPipedStream(context);
        try {
            obj.start();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        return obj.pipedInputStream;
    }
}
