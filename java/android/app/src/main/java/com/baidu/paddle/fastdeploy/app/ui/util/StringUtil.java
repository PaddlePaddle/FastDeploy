package com.baidu.paddle.fastdeploy.app.ui.util;

import java.text.DecimalFormat;

/**
 * Created by ruanshimin on 2018/5/24.
 */

public class StringUtil {
    public static String formatFloatString(float number) {
        DecimalFormat df = new DecimalFormat("0.00");
        return df.format(number);
    }
}
