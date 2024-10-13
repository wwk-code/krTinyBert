package com.k2fsa.sherpa.onnx;

import static com.k2fsa.sherpa.onnx.FeatureConfigKt.getFeatureConfig;
import static com.k2fsa.sherpa.onnx.OnlineRecognizerKt.getModelConfig;
import static com.k2fsa.sherpa.onnx.OnlineRecognizerKt.getOnlineLMConfig;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "sherpa-onnx";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;

    private final String[] permissions = {Manifest.permission.RECORD_AUDIO};

    private OnlineRecognizer recognizer;
    private AudioRecord audioRecord;
    private Button recordButton;
    private TextView textView;
    private ExecutorService recordingExecutor;

    private final int audioSource = MediaRecorder.AudioSource.MIC;
    private final int sampleRateInHz = 16000;
    private final int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    private final int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private int idx = 0;
    private String lastText = "";

    private volatile boolean isRecording = false;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        boolean permissionToRecordAccepted = requestCode == REQUEST_RECORD_AUDIO_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;

        if (!permissionToRecordAccepted) {
            Log.e(TAG, "Audio record is disallowed");
            finish();
        } else {
            Log.i(TAG, "Audio record is permitted");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);

        Log.i(TAG, "Start to initialize model");
        initModel();
        Log.i(TAG, "Finished initializing model");

        recordButton = findViewById(R.id.record_button);
        recordButton.setOnClickListener(v -> onclick());

        textView = findViewById(R.id.my_text);
        textView.setMovementMethod(new ScrollingMovementMethod());
    }

    private void onclick() {
        if (!isRecording) {
            boolean ret = initMicrophone();
            if (!ret) {
                Log.e(TAG, "Failed to initialize microphone");
                return;
            }
            Log.i(TAG, "state: " + (audioRecord != null ? audioRecord.getState() : "null"));
            audioRecord.startRecording();
            recordButton.setText(R.string.stop);
            isRecording = true;
            textView.setText("");
            lastText = "";
            idx = 0;

            recordingExecutor = Executors.newSingleThreadExecutor();
            recordingExecutor.execute(this::processSamples);
            Log.i(TAG, "Started recording");
        } else {
            isRecording = false;
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
            recordButton.setText(R.string.start);
            Log.i(TAG, "Stopped recording");
        }
    }

    private void processSamples() {
        Log.i(TAG, "processing samples");
        OnlineStream stream = recognizer.createStream("");

        double interval = 0.1; // i.e., 100 ms
        int bufferSize = (int) (interval * sampleRateInHz); // in samples
        short[] buffer = new short[bufferSize];

        while (isRecording) {
            int ret = audioRecord.read(buffer, 0, buffer.length);
            if (ret > 0) {
                float[] samples = new float[ret];
                for (int i = 0; i < ret; i++) {
                    samples[i] = buffer[i] / 32768.0f;
                }
                stream.acceptWaveform(samples, sampleRateInHz);
                while (recognizer.isReady(stream)) {
                    recognizer.decode(stream);
                }

                boolean isEndpoint = recognizer.isEndpoint(stream);
                String text = recognizer.getResult(stream).getText();

                if (isEndpoint && !recognizer.getConfig().getModelConfig().getParaformer().getEncoder().isEmpty()) {
                    float[] tailPaddings = new float[(int) (0.8 * sampleRateInHz)];
                    stream.acceptWaveform(tailPaddings, sampleRateInHz);
                    while (recognizer.isReady(stream)) {
                        recognizer.decode(stream);
                    }
                    text = recognizer.getResult(stream).getText();
                }

                String textToDisplay = lastText;

                if (!text.isEmpty()) {
                    textToDisplay = lastText.isEmpty() ? idx + ": " + text : lastText + "\n" + idx + ": " + text;
                }

                if (isEndpoint) {
                    recognizer.reset(stream);
                    if (!text.isEmpty()) {
                        lastText += "\n" + idx + ": " + text;
                        textToDisplay = lastText;
                        idx++;
                    }
                }
                final String displayText = textToDisplay;
                runOnUiThread(() -> textView.setText(displayText));
            }
        }
        stream.release();
    }

    private boolean initMicrophone() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
            return false;
        }

        int numBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat);
        Log.i(TAG, "buffer size in milliseconds: " + (numBytes * 1000.0f / sampleRateInHz));

        audioRecord = new AudioRecord(audioSource, sampleRateInHz, channelConfig, audioFormat, numBytes * 2);
        return true;
    }

    private void initModel() {
        int type = 5;
        String ruleFsts = null;

        Log.i(TAG, "Select model type " + type);

        OnlineRecognizerConfig config = new OnlineRecognizerConfig(
                getFeatureConfig(sampleRateInHz, 80),
                getModelConfig(type),
                getOnlineLMConfig(type),
                new OnlineCtcFstDecoderConfig("",3000),
                new EndpointConfig(new EndpointRule(false, 2.4f, 0.0f),new EndpointRule(true, 1.4f, 0.0f), new EndpointRule(false, 0.0f, 20.0f)),
                true,
                "greedy_search",
                4,
                "",
                1.5f,
                "",
                "",
                0.0f
        );

        if (ruleFsts != null) {
            config.setRuleFsts(ruleFsts);
        }

        recognizer = new OnlineRecognizer(getAssets(), config);
    }
}