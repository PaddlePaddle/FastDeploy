package com.baidu.aip.asrwakeup3.core.util.bluetooth;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.media.AudioManager;
import android.util.Log;

public class HeadsetReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        String action = intent.getAction();
        if (action.equals(AudioManager.ACTION_HEADSET_PLUG)) {
            // This happens when the user plugs a Jack headset to the device for example
            int state = intent.getIntExtra("state", 0);
            String name = intent.getStringExtra("name");
            int hasMicrophone = intent.getIntExtra("microphone", 0);

            if (state == 0) {
                Log.i("HeadsetReceiver", "[Headset] Headset disconnected:" + name);
            } else if (state == 1) {
                Log.i("HeadsetReceiver", "[Headset] Headset connected:" + name);
                if (hasMicrophone == 1) {
                    Log.i("HeadsetReceiver", "[Headset] Headset " + name + " has a microphone");
                }
            } else {
                Log.w("HeadsetReceiver", "[Headset] Unknown headset plugged state: " + state);
            }

            AndroidAudioManager.getInstance(context).routeAudioToEarPiece();
            // LinphoneManager.getCallManager().refreshInCallActions();
        } else if (action.equals(AudioManager.ACTION_AUDIO_BECOMING_NOISY)) {
            // This happens when the user disconnect a headset, so we shouldn't play audio loudly
            Log.i("HeadsetReceiver", "[Headset] Noisy state detected, most probably a headset has been disconnected");
            AndroidAudioManager.getInstance(context).routeAudioToEarPiece();
            // LinphoneManager.getCallManager().refreshInCallActions();
        } else {
            Log.w("HeadsetReceiver", "[Headset] Unknown action: " + action);
        }
    }
}