package com.baidu.aip.asrwakeup3.core.util.bluetooth;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothHeadset;
import android.bluetooth.BluetoothProfile;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.AudioManager;
import android.util.Log;

import java.util.List;

import static android.media.AudioManager.STREAM_VOICE_CALL;

public class AndroidAudioManager {

    private static volatile AndroidAudioManager instance;

    private BluetoothAdapter mBluetoothAdapter;

    private AudioManager mAudioManager;

    private boolean mIsBluetoothHeadsetConnected;

    private boolean mIsBluetoothHeadsetScoConnected;

    private BluetoothReceiver mBluetoothReceiver;

    private HeadsetReceiver mHeadsetReceiver;

    private boolean mAudioFocused;

    private Context mContext;
    private BluetoothHeadset mBluetoothHeadset;

    private AndroidAudioManager(Context context) {
        mAudioManager = ((AudioManager) context.getSystemService(Context.AUDIO_SERVICE));
        this.mContext = context.getApplicationContext();
    }

    public AudioManager getAudioManager() {
        return mAudioManager;
    }

    public static AndroidAudioManager getInstance(Context context) {
        if (instance == null) {
            synchronized (AndroidAudioManager.class) {
                if (instance == null) {
                    instance = new AndroidAudioManager(context);
                }
            }
        }
        return instance;
    }

    public void startBluetooth() {
        mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
        if (mBluetoothAdapter != null) {
            Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] Adapter found");
            if (mAudioManager.isBluetoothScoAvailableOffCall()) {
                Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] SCO available off call, continue");
            } else {
                Log.w("AndroidAudioManager", "[Audio Manager] [Bluetooth] SCO not available off call !");
            }
            if (mBluetoothAdapter.isEnabled()) {
                Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] Adapter enabled");
                mBluetoothReceiver = new BluetoothReceiver();
                mIsBluetoothHeadsetConnected = false;
                mIsBluetoothHeadsetScoConnected = false;

                BluetoothProfile.ServiceListener bluetoothServiceListener =
                        new BluetoothProfile.ServiceListener() {
                            public void onServiceConnected(int profile, BluetoothProfile proxy) {
                                if (profile == BluetoothProfile.HEADSET) {
                                    Log.i("AndroidAudioManager",
                                            "[Audio Manager] [Bluetooth] HEADSET profile connected");
                                    mBluetoothHeadset = (BluetoothHeadset) proxy;

                                    List<BluetoothDevice> devices =
                                            mBluetoothHeadset.getConnectedDevices();
                                    if (devices.size() > 0) {
                                        Log.i("AndroidAudioManager",
                                                "[Audio Manager] [Bluetooth] A device is already connected");
                                        bluetoothHeadetConnectionChanged(true);
                                    }

                                    Log.i("AndroidAudioManager",
                                            "[Audio Manager] [Bluetooth] Registering bluetooth receiver");

                                    IntentFilter filter = new IntentFilter();
                                    filter.addAction(BluetoothHeadset.ACTION_AUDIO_STATE_CHANGED);
                                    filter.addAction(
                                            BluetoothHeadset.ACTION_CONNECTION_STATE_CHANGED);
                                    filter.addAction(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED);
                                    filter.addAction(
                                            BluetoothHeadset.ACTION_VENDOR_SPECIFIC_HEADSET_EVENT);

                                    Intent sticky =
                                            mContext.registerReceiver(mBluetoothReceiver, filter);
                                    int state =
                                            sticky.getIntExtra(
                                                    AudioManager.EXTRA_SCO_AUDIO_STATE,
                                                    AudioManager.SCO_AUDIO_STATE_DISCONNECTED);
                                    if (state == AudioManager.SCO_AUDIO_STATE_CONNECTED) {
                                        Log.i("AndroidAudioManager",
                                                "[Audio Manager] [Bluetooth] Bluetooth headset SCO connected");
                                        bluetoothHeadetScoConnectionChanged(true);
                                    } else if (state == AudioManager.SCO_AUDIO_STATE_DISCONNECTED) {
                                        Log.i("AndroidAudioManager",
                                                "[Audio Manager] [Bluetooth] Bluetooth headset SCO disconnected");
                                        bluetoothHeadetScoConnectionChanged(false);
                                    } else if (state == AudioManager.SCO_AUDIO_STATE_CONNECTING) {
                                        Log.i("AndroidAudioManager",
                                                "[Audio Manager] [Bluetooth] Bluetooth headset SCO connecting");
                                    } else if (state == AudioManager.SCO_AUDIO_STATE_ERROR) {
                                        Log.i("AndroidAudioManager",
                                                "[Audio Manager] [Bluetooth] Bluetooth headset SCO connection error");
                                    } else {
                                        Log.w("AndroidAudioManager",
                                                "[Audio Manager] [Bluetooth] Bluetooth headset " +
                                                        "unknown SCO state changed: "
                                                        + state);
                                    }
                                }
                            }

                            public void onServiceDisconnected(int profile) {
                                if (profile == BluetoothProfile.HEADSET) {
                                    Log.i("AndroidAudioManager",
                                            "[Audio Manager] [Bluetooth] HEADSET profile disconnected");
                                    mBluetoothHeadset = null;
                                    mIsBluetoothHeadsetConnected = false;
                                    mIsBluetoothHeadsetScoConnected = false;
                                }
                            }
                        };
                mBluetoothAdapter.getProfileProxy(
                        mContext, bluetoothServiceListener, BluetoothProfile.HEADSET);
            }
        }
    }


    // Bluetooth

    public synchronized void bluetoothHeadetConnectionChanged(boolean connected) {
        mIsBluetoothHeadsetConnected = connected;
        mAudioManager.setBluetoothScoOn(connected);
        mAudioManager.startBluetoothSco();
        routeAudioToBluetooth();
    }


    public synchronized boolean isBluetoothHeadsetConnected() {
        return mIsBluetoothHeadsetConnected;
    }

    public synchronized void bluetoothHeadetScoConnectionChanged(boolean connected) {
        mIsBluetoothHeadsetScoConnected = connected;
    }

    public synchronized boolean isUsingBluetoothAudioRoute() {
        return mIsBluetoothHeadsetScoConnected;
    }

    public synchronized void routeAudioToBluetooth() {
        if (!isBluetoothHeadsetConnected()) {
            Log.w("AndroidAudioManager", "[Audio Manager] [Bluetooth] No headset connected");
            return;
        }
        if (mAudioManager.getMode() != AudioManager.MODE_IN_COMMUNICATION) {
            Log.w("AndroidAudioManager",
                    "[Audio Manager] [Bluetooth] Changing audio mode to MODE_IN_COMMUNICATION " +
                            "and requesting STREAM_VOICE_CALL focus");
            mAudioManager.setMode(AudioManager.MODE_IN_COMMUNICATION);
            requestAudioFocus(STREAM_VOICE_CALL);
        }
        changeBluetoothSco(true);
    }

    private void requestAudioFocus(int stream) {
        if (!mAudioFocused) {
            int res =
                    mAudioManager.requestAudioFocus(
                            null, stream, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE);
            Log.d("AndroidAudioManager",
                    "[Audio Manager] Audio focus requested: "
                            + (res == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
                            ? "Granted"
                            : "Denied"));
            if (res == AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                mAudioFocused = true;
            }
        }
    }

    private synchronized void changeBluetoothSco(final boolean enable) {
        // IT WILL TAKE A CERTAIN NUMBER OF CALLS TO EITHER START/STOP BLUETOOTH SCO FOR IT TO WORK
        if (enable && mIsBluetoothHeadsetScoConnected) {
            Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] SCO already enabled, skipping");
            return;
        } else if (!enable && !mIsBluetoothHeadsetScoConnected) {
            Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] SCO already disabled, skipping");
            return;
        }

        new Thread() {
            @Override
            public void run() {
                boolean resultAcknoledged;
                int retries = 0;
                do {
                    try {
                        Thread.sleep(200);
                    } catch (InterruptedException e) {
                        Log.e("AndroidAudioManager", e.getMessage(), e);
                    }

                    synchronized (AndroidAudioManager.this) {
                        if (enable) {
                            Log.i("AndroidAudioManager",
                                    "[Audio Manager] [Bluetooth] Starting SCO: try number "
                                            + retries);
                            mAudioManager.startBluetoothSco();
                        } else {
                            Log.i("AndroidAudioManager",
                                    "[Audio Manager] [Bluetooth] Stopping SCO: try number "
                                            + retries);
                            mAudioManager.stopBluetoothSco();
                        }
                        resultAcknoledged = isUsingBluetoothAudioRoute() == enable;
                        retries++;
                    }
                } while (!resultAcknoledged && retries < 10);
            }
        }.start();
    }

    public void destroy() {
        if (mBluetoothAdapter != null && mBluetoothHeadset != null) {
            Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] Closing HEADSET profile proxy");
            mBluetoothAdapter.closeProfileProxy(BluetoothProfile.HEADSET, mBluetoothHeadset);
        }

        Log.i("AndroidAudioManager", "[Audio Manager] [Bluetooth] Unegistering bluetooth receiver");
        if (mBluetoothReceiver != null) {
            mContext.unregisterReceiver(mBluetoothReceiver);
        }
        synchronized (AndroidAudioManager.class) {
            mContext = null;
            instance = null;
        }
    }


    public void startSimpleBluetooth() {
        mAudioManager.setBluetoothScoOn(true);
        mAudioManager.startBluetoothSco();
    }

    public void destorySimpleBluetooth() {
        mAudioManager.setBluetoothScoOn(false);
        mAudioManager.stopBluetoothSco();
    }

    // HEADSET 插耳机的

    public void enableHeadsetReceiver() {
        mHeadsetReceiver = new HeadsetReceiver();

        Log.i("AndroidAudioManager", "[Audio Manager] Registering headset receiver");
        mContext.registerReceiver(
                mHeadsetReceiver, new IntentFilter(AudioManager.ACTION_AUDIO_BECOMING_NOISY));
        mContext.registerReceiver(
                mHeadsetReceiver, new IntentFilter(AudioManager.ACTION_HEADSET_PLUG));
    }

    public void routeAudioToEarPiece() {
        routeAudioToSpeakerHelper(false);
    }

    public void routeAudioToSpeakerHelper(boolean speakerOn) {
        Log.w("AndroidAudioManager", "[Audio Manager] Routing audio to " + (speakerOn ? "speaker" : "earpiece"));
        if (mIsBluetoothHeadsetScoConnected) {
            Log.w("AndroidAudioManager", "[Audio Manager] [Bluetooth] Disabling bluetooth audio route");
            changeBluetoothSco(false);
        }

        mAudioManager.setSpeakerphoneOn(speakerOn);
    }
}
