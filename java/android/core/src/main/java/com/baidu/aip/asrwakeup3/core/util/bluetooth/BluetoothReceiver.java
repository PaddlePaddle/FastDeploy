package com.baidu.aip.asrwakeup3.core.util.bluetooth;

import android.bluetooth.BluetoothHeadset;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.media.AudioManager;
import android.util.Log;

public class BluetoothReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        String action = intent.getAction();
        if (action.equals(BluetoothHeadset.ACTION_CONNECTION_STATE_CHANGED)) {
            int state =
                    intent.getIntExtra(
                            BluetoothHeadset.EXTRA_STATE, BluetoothHeadset.STATE_DISCONNECTED);
            if (state == BluetoothHeadset.STATE_CONNECTED) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset connected");
                AndroidAudioManager.getInstance(context).bluetoothHeadetConnectionChanged(true);
            } else if (state == BluetoothHeadset.STATE_DISCONNECTED) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset disconnected");
                AndroidAudioManager.getInstance(context).bluetoothHeadetConnectionChanged(false);
            } else {
                Log.w("BluetoothReceiver", "[Bluetooth] Bluetooth headset unknown state changed: " + state);
            }
        } else if (action.equals(BluetoothHeadset.ACTION_AUDIO_STATE_CHANGED)) {
            int state =
                    intent.getIntExtra(
                            BluetoothHeadset.EXTRA_STATE,
                            BluetoothHeadset.STATE_AUDIO_DISCONNECTED);
            if (state == BluetoothHeadset.STATE_AUDIO_CONNECTED) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset audio connected");
                // AndroidAudioManager.getInstance(context).bluetoothHeadetAudioConnectionChanged(true);
            } else if (state == BluetoothHeadset.STATE_AUDIO_DISCONNECTED) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset audio disconnected");
                // AndroidAudioManager.getInstance(context).bluetoothHeadetAudioConnectionChanged(false);
            } else {
                Log.w("BluetoothReceiver", "[Bluetooth] Bluetooth headset unknown audio state changed: " + state);
            }
        } else if (action.equals(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED)) {
            int state =
                    intent.getIntExtra(
                            AudioManager.EXTRA_SCO_AUDIO_STATE,
                            AudioManager.SCO_AUDIO_STATE_DISCONNECTED);
            if (state == AudioManager.SCO_AUDIO_STATE_CONNECTED) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset SCO connected");
                AndroidAudioManager.getInstance(context).bluetoothHeadetScoConnectionChanged(true);
            } else if (state == AudioManager.SCO_AUDIO_STATE_DISCONNECTED) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset SCO disconnected");
                AndroidAudioManager.getInstance(context).bluetoothHeadetScoConnectionChanged(false);
            } else if (state == AudioManager.SCO_AUDIO_STATE_CONNECTING) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset SCO connecting");
            } else if (state == AudioManager.SCO_AUDIO_STATE_ERROR) {
                Log.i("BluetoothReceiver", "[Bluetooth] Bluetooth headset SCO connection error");
            } else {
                Log.w("BluetoothReceiver", "[Bluetooth] Bluetooth headset unknown SCO state changed: " + state);
            }
        } else if (action.equals(BluetoothHeadset.ACTION_VENDOR_SPECIFIC_HEADSET_EVENT)) {
            String command =
                    intent.getStringExtra(BluetoothHeadset.EXTRA_VENDOR_SPECIFIC_HEADSET_EVENT_CMD);
            int type =
                    intent.getIntExtra(
                            BluetoothHeadset.EXTRA_VENDOR_SPECIFIC_HEADSET_EVENT_CMD_TYPE, -1);

            String commandType;
            switch (type) {
                case BluetoothHeadset.AT_CMD_TYPE_ACTION:
                    commandType = "AT Action";
                    break;
                case BluetoothHeadset.AT_CMD_TYPE_READ:
                    commandType = "AT Read";
                    break;
                case BluetoothHeadset.AT_CMD_TYPE_TEST:
                    commandType = "AT Test";
                    break;
                case BluetoothHeadset.AT_CMD_TYPE_SET:
                    commandType = "AT Set";
                    break;
                case BluetoothHeadset.AT_CMD_TYPE_BASIC:
                    commandType = "AT Basic";
                    break;
                default:
                    commandType = "AT Unknown";
                    break;
            }
            Log.i("BluetoothReceiver", "[Bluetooth] Vendor action " + commandType + " : " + command);
        } else {
            Log.w("BluetoothReceiver", "[Bluetooth] Bluetooth unknown action: " + action);
        }
    }
}