package com.qualcomm.qti.sentimentanalysis;

import static com.qualcomm.qti.sentimentanalysis.tinyBERT.tinyBERT.readInputExamples;
import static com.qualcomm.qti.sentimentanalysis.tinyBERT.tinyBERT.readMultipleInputExamples;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.qualcomm.qti.sentimentanalysis.databinding.ActivityMainBinding;
import com.qualcomm.qti.sentimentanalysis.tinyBERT.Tokenization;
import com.qualcomm.qti.sentimentanalysis.tinyBERT.tinyBERT;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.LongStream;

import com.qualcomm.qti.sentimentanalysis.tokenization.FullTokenizer;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'sentimentanalysis' library on application startup.
    static {
        System.loadLibrary("sentimentanalysis");
    }

    private ActivityMainBinding binding;
    private AssetManager assetManager;

    private String prevResult = "";
    private String result = "";
    private static final String TAG = "SNPE_SA";
    private static final String BENCHMARK_TAG = "BENCHMARK";
    private static final String DIC_PATH = "vocab.txt";
    private static final int MAX_SEQ_LEN = 128;
    private static final boolean DO_LOWER_CASE = true;
    private final Map<String, Integer> inputDic = new HashMap<>();
    private FullTokenizer tokenizer;
    private String DEBUG_TAG = "TEST";
    private static final int REQUEST_CODE_WRITE_EXTERNAL_STORAGE = 1;

    public synchronized void loadDictionary() {
        try {
            Log.v(TAG, "==> Loading Dictionary .");
            loadDictionaryFile(this.getAssets());
            Log.v(TAG, "Dictionary loaded.");
        } catch (IOException ex) {
            Log.e(TAG, "Dictionary load exception:" + ex.getMessage());
        }
    }

    /** Load dictionary from assets. */
    public void loadDictionaryFile(AssetManager assetManager) throws IOException {
        try (InputStream ins = assetManager.open(DIC_PATH);
             BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
            int index = 0;
            while (reader.ready()) {
                String key = reader.readLine();
                inputDic.put(key, index++);
            }
        }
    }

    public List<Integer> preProcessor(String sentence) {
        List<String> tokens = new ArrayList<>();
        // Start of generating the features.
        tokens.add("[CLS]");
        tokens = tokenizer.tokenize(sentence);
        // For ending mark.
        tokens.add("[SEP]");

        return tokenizer.convertTokensToIds(tokens);
    }

    public String formatOutput(String txtOutput, long infTime,
                               boolean isLive, String runtime, String... userInput) {
        String setOutput;
        String[] res = txtOutput.split("\\s+");
        Integer positive = Integer.parseInt(res[0]);
        Integer negative = Integer.parseInt(res[1]) + 1;
        Log.i("SNPE_INF","positive : "+positive.toString()+ "\t negative: " + negative.toString());
        Log.i(TAG, "INF time = " + infTime);

        if (isLive) {
            setOutput = "\nPositivity : "+positive.toString()+ " %  \t\t  " +
                    "Negativity: " + negative.toString() + " %" ;
            setOutput += "\n" + runtime + " Exec Time = " + infTime + "ms";
            return setOutput;
        }

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
        LocalDateTime now = LocalDateTime.now();
        result = "Result at time: "+dtf.format(now)+"\n\n" +"input: "+ Arrays.toString(userInput) +
                "\nOutput: Positivity : "+positive.toString()+ " %  \t\t  " +
                "Negativity: " + negative.toString() + " %" ;
        result += "\n" + runtime + " Exec Time = " + infTime + "ms";
        setOutput = "__________________________________________\n"+
                " SA Predicted Result \n__________________________________________\n"+
                result +"\n"+ prevResult.toString();
        return setOutput;
    }


    public void checkLibraryFiles() {
        String nativeLibraryDir = getApplicationInfo().nativeLibraryDir;
        File libDir = new File(nativeLibraryDir);
        if (libDir.isDirectory()) {
            String[] fileList = libDir.list();
            if (fileList != null) { // 确保 fileList 不为 null
                for (String file : fileList) {
                    // 打印文件名
                    Log.i(DEBUG_TAG, "Found library file: " + file);
                }
            } else {
                Log.d(DEBUG_TAG, "No files found in the directory.");
            }
        } else {
            Log.d(DEBUG_TAG, "The specified path is not a directory.");
        }
    }

    // fileName单给: output.txt,大概率存储在手机文件系统的 /storage/emulated/0/Documents/output.txt 下， 0 可能需要换成别的数字
    public static void writeToExternalFile(Context context, String fileName, float[] data) {
        // 检查权限
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            // 权限未授予，返回
            return;
        }

        // 获取外部存储目录
        File externalDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
        File file = new File(externalDir, fileName);
        FileOutputStream fos = null;
        try {
            // 创建父目录（如果不存在）
            if (!externalDir.exists()) {
                externalDir.mkdirs();
            }
            fos = new FileOutputStream(file);
            // 写入 float 数组到文件
            for (float value : data) {
                fos.write((value + "\n").getBytes()); // 每个值后加换行符
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        tinyBERT tinybert = new tinyBERT(this);
        Tokenization tokenization = new Tokenization();
        Tokenization.BertTokenizer bertTokenizer;
//        String runtime = "CPU";
        String runtime = "DSP";
        assetManager = getAssets();
        // 注意java中实例化非静态内部类的语法
        tinyBERT.Sst2Processor sst2Processor = tinybert.new Sst2Processor(this);
        try {
            bertTokenizer = tokenization.getBertTokenizer(this);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        boolean ret = exportDSPENV(getApplicationInfo().nativeLibraryDir);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        String nativeDirPath = getApplicationInfo().nativeLibraryDir;
        String uiLogger = "";
        TextView tv = binding.textView;
        Button warmUpButton = findViewById(R.id.button);
        Button benchMarkButton = findViewById(R.id.button2);

        // 检查app读写文件权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            // 请求权限
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CODE_WRITE_EXTERNAL_STORAGE);
        } else {
            // 权限已被授予，您可以执行写文件的操作
            System.out.println("temp");
        }

        // init SNPE
        assetManager = getAssets();
        Log.i(TAG, "onCreate: Initializing SNPE ...");
        uiLogger += initSNPE(assetManager);
        tv.setText(uiLogger);
        initSNPE(assetManager);

        int dataBatch = 100;
        int benchMarkTimes = 20;
        int inputDataLen = 128;   // 目前三个输入参数数组长度均为64
        int[] arrayLengths = {inputDataLen, inputDataLen};
        List<List<int[]>> inputDatas = readMultipleInputExamples(MainActivity.this);
        warmUpButton.setOnClickListener(
                (View v) -> {
                    for (int i = 0; i < dataBatch; i++) {
                        float[] output1 = inferSNPE(runtime, inputDatas.get(i).get(0), inputDatas.get(i).get(1), arrayLengths, 1);
                        writeToExternalFile(this,"output_" + String.valueOf(i) + ".txt",output1);
                        System.out.println(" ");
                    }
                    String showText = "Generate output files!";
                    Log.i(BENCHMARK_TAG,showText);
                    tv.setText(showText);
                });
//
//        benchMarkButton.setOnClickListener(
//                (View v) -> {
//                    inferSNPE(runtime, inputDatas.get(0), inputDatas.get(1), inputDatas.get(2), arrayLengths, benchMarkTimes);
//                    String showText = "Benchmark " + benchMarkTimes + " times finished!";
//                    Log.i(BENCHMARK_TAG,showText);
//                    tv.setText(showText);
//                });
    }

    /**
     * A native method that is implemented by the 'sentimentanalysis' native library,
     * which is packaged with this application.
     */
    public native String queryRuntimes(String nativeDirPath);
    public native String initSNPE(AssetManager assetManager);
    public native float[] inferSNPE(String runtime, int[] input_ids, int[] attention_mask, int[] arrayLengths, int executeTimes);
    public native boolean exportDSPENV(String path);
    public native String getEnv(String varName);
}