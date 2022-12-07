package com.baidu.aip.asrwakeup3.core.inputstream;

import android.util.Log;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Created by fujiayi on 2017/11/27.
 * <p>
 * 解决大文件的输入问题。
 * 文件大时不能通过Infile参数一下子输入。
 */

public class FileAudioInputStream extends InputStream {

    private InputStream in;

    private long nextSleepTime = -1;

    private long totalSleepMs = 0;

    private static final String TAG = "FileAudioInputStream";

    public FileAudioInputStream(String file) throws FileNotFoundException {
        in = new FileInputStream(file);
    }

    public FileAudioInputStream(InputStream in) {
        this.in = in;
    }

    @Override
    public int read() throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public int read(byte[] buffer, int byteOffset, int byteCount) throws IOException {
        int bytePerMs = 16000 * 2 / 1000;
        int count = bytePerMs * 20; // 20ms 音频数据
        if (byteCount < count) {
            count = byteCount;
        }
        if (nextSleepTime > 0) {
            try {
                long sleepMs = nextSleepTime - System.currentTimeMillis();
                if (sleepMs > 0) {
                    Log.i(TAG, "will sleep " + sleepMs);
                    Thread.sleep(sleepMs); // 每20ms的音频 ，比如等待20ms传输下一批
                    totalSleepMs += sleepMs;
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        int r = in.read(buffer, byteOffset, count);

        /*
        if (r >= 0) {
            Log.i("FileAudioInputStream", "Debug:" + System.currentTimeMillis() + ": " + md5(buffer, byteOffset, r));
        } else {
            Log.i("FileAudioInputStream", "Debug:" + System.currentTimeMillis() + ": return " + r);
        }
        */
        nextSleepTime = System.currentTimeMillis() + r / bytePerMs;

        // 如果是长语音，在r=-1的情况下，需要手动调用stop
        return r;
    }

    @Override
    public void close() throws IOException {
        super.close();
        Log.i(TAG, "time sleeped " + totalSleepMs);
        if (null != in) {
            in.close();
        }
    }

    private String md5(byte[] buffer, int byteOffset, int byteCount) {
        try {
            MessageDigest digest = MessageDigest.getInstance("MD5");
            digest.reset();
            digest.update(buffer, byteOffset, byteCount);
            BigInteger bigInt = new BigInteger(1, digest.digest());
            return bigInt.toString(16);
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return null;
    }
}
