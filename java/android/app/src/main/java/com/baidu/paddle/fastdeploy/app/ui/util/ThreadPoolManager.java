/*
 * Copyright (C) 2017 Baidu, Inc. All Rights Reserved.
 */
package com.baidu.paddle.fastdeploy.app.ui.util;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadPoolManager {

    static Timer timerFocus = null;

    /*
     * 对焦频率
     */
    static final long cameraScanInterval = 2000;

    /*
     * 线程池大小
     */
    private static int poolCount = Runtime.getRuntime().availableProcessors();

    private static ExecutorService fixedThreadPool = Executors.newFixedThreadPool(poolCount);

    private static ExecutorService singleThreadPool = Executors.newSingleThreadExecutor();

    /**
     * 给线程池添加任务
     *
     * @param runnable 任务
     */
    public static void execute(Runnable runnable) {
        fixedThreadPool.execute(runnable);
    }

    /**
     * 单独线程任务
     *
     * @param runnable 任务
     */
    public static void executeSingle(Runnable runnable) {
        singleThreadPool.execute(runnable);
    }

    /**
     * 创建一个定时对焦的timer任务
     *
     * @param runnable 对焦代码
     * @return Timer Timer对象，用来终止自动对焦
     */
    public static Timer createAutoFocusTimerTask(final Runnable runnable) {
        if (timerFocus != null) {
            return timerFocus;
        }
        timerFocus = new Timer();
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
                runnable.run();
            }
        };
        timerFocus.scheduleAtFixedRate(task, 0, cameraScanInterval);
        return timerFocus;
    }

    /**
     * 终止自动对焦任务，实际调用了cancel方法并且清空对象
     * 但是无法终止执行中的任务，需额外处理
     */
    public static void cancelAutoFocusTimer() {
        if (timerFocus != null) {
            timerFocus.cancel();
            timerFocus = null;
        }
    }
}
