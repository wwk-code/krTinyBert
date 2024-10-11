package com.qualcomm.qti.sentimentanalysis.tinyBERT;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;


public class tinyBERT {

    public Context context;
    public int maxSeqLength = 128;
    public Tokenization.BertTokenizer bertTokenizer;

    public int getMaxSeqLength() {
        return this.maxSeqLength;
    }

    public Tokenization.BertTokenizer getBertTokenizer() {
        return this.bertTokenizer;
    }

    public tinyBERT(Context context) {
        this.context = context;
    }

    public tinyBERT(Context context,Tokenization.BertTokenizer bertTokenizer) {
        this.context = context;
        this.bertTokenizer = bertTokenizer;
    }

    public class InputFeatures {
        /** A single set of features of data. */
        private int[] inputIds;
        private int[] inputMask;
        private int[] segmentIds;
        private Integer seqLength; // 使用 Integer 以支持 null 值
        private Integer labelId;

        // 构造函数
        public InputFeatures(int[] inputIds, int[] inputMask, int[] segmentIds, Integer labelId, Integer seqLength) {
            this.inputIds = inputIds;
            this.inputMask = inputMask;
            this.segmentIds = segmentIds;
            this.labelId = labelId;
            this.seqLength = seqLength;
        }

        // Getter 方法
        public int[] getInputIds() {
            return inputIds;
        }

        public int[] getInputMask() {
            return inputMask;
        }

        public int[] getSegmentIds() {
            return segmentIds;
        }

        public Integer getSeqLength() {
            return seqLength;
        }

        public Integer getLabelId() {
            return labelId;
        }

    }


    public class InputExample {
        /** A single training/test example for simple sequence classification. */

        private String guid;
        private String textA;
        private String textB; // 可选
        private String label;  // 可选

        /** Constructs an InputExample. */
        public InputExample(String guid, String textA, String textB, String label) {
            this.guid = guid;
            this.textA = textA;
            this.textB = textB;
            this.label = label;
        }

        // Getter 方法
        public String getGuid() {
            return guid;
        }

        public String getTextA() {
            return textA;
        }

        public String getTextB() {
            return textB;
        }

        public String getLabel() {
            return label;
        }
    }

    public class TsvReader {

        private Context context;

        public TsvReader(Context context) {
            this.context = context;
        }

        public List<List<String>> readTsvFile(String fileName) {
            AssetManager assetManager = context.getAssets();
            List<List<String>> lines = new ArrayList<>();
            try (InputStream inputStream = assetManager.open(fileName);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    // 按制表符分隔每一行
                    String[] values = line.split("\t");
                    List<String> row = new ArrayList<>();
                    for (String value : values) {
                        row.add(value);
                    }
                    lines.add(row);
                }
            } catch (IOException e) {
                e.printStackTrace(); // 处理异常，可以根据需要改为抛出异常
            }

            return lines;
        }

    }


    // 处理 SST-2 数据集的类
    public class Sst2Processor{
        /** Processor for the SST-2 data set (GLUE version). */
        public Context context;
        public TsvReader tsvReader;
        public Sst2Processor(Context context) {
            this.context = context;
            tsvReader = new TsvReader(this.context);
        }

        public List<String> getLables() {
            List<String> lableList = new ArrayList<>();
            lableList.add("0");
            lableList.add("1");
            return lableList;
        }

        public List<InputExample> getDevExamples(String filename) throws IOException {
            // See base class
            return createExamples(tsvReader.readTsvFile(filename),"dev");
        }

        public List<String> getLabels() {
            // See base class
            List<String> labels = new ArrayList<>();
            labels.add("0");
            labels.add("1");
            return labels;
        }

        private List<InputExample> createExamples(List<List<String>> lines, String setType) {
            /** Creates examples for the training and dev sets. */
            List<InputExample> examples = new ArrayList<>();
            for (int i = 0; i < lines.size(); i++) {
                if (i == 0) {
                    continue; // Skip the header
                }
                String guid = setType + "-" + i;
                String textA = lines.get(i).get(0);
                String label = lines.get(i).get(1);
                examples.add(new InputExample(guid, textA, null, label));
            }
            return examples;
        }

    }

    public void truncateSeqPair(List<String> tokensA, List<String> tokensB, int maxLength) {
        /** Truncates a sequence pair in place to the maximum length. */
        while (true) {
            int totalLength = tokensA.size() + tokensB.size();
            if (totalLength <= maxLength) {
                break;
            }
            if (tokensA.size() > tokensB.size()) {
                tokensA.remove(0); // 移除第一个元素
            } else {
                tokensB.remove(0); // 移除第一个元素
            }
        }
    }

    public List<InputFeatures> convertExamplesToFeatures(List<InputExample> examples,
                                                         List<String> labelList, int maxSeqLength, Tokenization.BertTokenizer tokenizer, String outputMode) {
        /** Loads a data file into a list of `InputBatch`s. */
        Map<String, Integer> labelMap = new HashMap<>();
        for (int i = 0; i < labelList.size(); i++) {
            labelMap.put(labelList.get(i), i);
        }

        List<InputFeatures> features = new ArrayList<>();
        for (int exIndex = 0; exIndex < examples.size(); exIndex++) {
            InputExample example = examples.get(exIndex);

            List<String> tokensA = tokenizer.tokenize(example.getTextA());
            List<String> tokensB = null;
            if (example.getTextB() != null) {
                tokensB = tokenizer.tokenize(example.getTextB());
                truncateSeqPair(tokensA, tokensB, maxSeqLength - 3);
            } else {
                if (tokensA.size() > maxSeqLength - 2) {
                    tokensA = tokensA.subList(0, maxSeqLength - 2);
                }
            }
            List<String> tokens = new ArrayList<>();
            tokens.add("[CLS]");
            tokens.addAll(tokensA);
            tokens.add("[SEP]");

            List<Integer> segmentIds = new ArrayList<>();
            for (int i = 0; i < tokens.size(); i++) {
                segmentIds.add(0);
            }

            if (tokensB != null) {
                tokens.addAll(tokensB);
                tokens.add("[SEP]");
                for (int i = 0; i < tokensB.size() + 1; i++) {
                    segmentIds.add(1);
                }
            }

            List<Integer> inputIds = tokenizer.convertTokensToIds(Tokenization.readVocabFile("vocab.txt",this.context) ,tokens);
            List<Integer> inputMask = new ArrayList<>();
            for (int i = 0; i < inputIds.size(); i++) {
                inputMask.add(1);
            }
            int seqLength = inputIds.size();

            // Padding mechanism
            while (inputIds.size() < maxSeqLength) {
                inputIds.add(0); // Padding
            }
            while (inputMask.size() < maxSeqLength) {
                inputMask.add(0); // Padding
            }
            while (segmentIds.size() < maxSeqLength) {
                segmentIds.add(0); // Padding
            }

            // Assertions
            assert inputIds.size() == maxSeqLength;
            assert inputMask.size() == maxSeqLength;
            assert segmentIds.size() == maxSeqLength;

            int labelId;
            if (outputMode.equals("classification")) {
                labelId = labelMap.get(example.getLabel());
            } else if (outputMode.equals("regression")) {
                labelId = (int) Float.parseFloat(example.getLabel());
            } else {
                throw new IllegalArgumentException("Invalid output mode: " + outputMode);
            }
            features.add(new InputFeatures(inputIds.stream().mapToInt(i -> i).toArray(),
                    inputMask.stream().mapToInt(i -> i).toArray(),
                    segmentIds.stream().mapToInt(i -> i).toArray(),
                    labelId,
                    seqLength));
        }
        return features;
    }

    // 从 Features 中获取到tinyBERT模型的有效数据
    public static Map<String,Object> getTensorDatas(List<InputFeatures> features) {
        Map<String,Object> tensorDatasMap = new HashMap<>();
        List<Integer> all_label_ids = new ArrayList<>();
        List<Integer> all_seq_lengths = new ArrayList<>();
        List<List<Long>> all_input_ids = new ArrayList<>();
        List<List<Long>> all_input_mask = new ArrayList<>();
        List<List<Long>> all_segment_ids = new ArrayList<>();
        int lenFeatures = features.size();
        for (int i = 0; i < lenFeatures; i++) {
            all_label_ids.add(features.get(i).getLabelId());
            all_seq_lengths.add(features.get(i).getSeqLength());
            int truncateLen = 64;
            all_input_ids.add(Arrays.stream(features.get(i).getInputIds()).mapToLong(j -> (long)j).boxed().collect(Collectors.toList()).subList(0,truncateLen));
            all_input_mask.add(Arrays.stream(features.get(i).getInputMask()).mapToLong(j -> (long)j).boxed().collect(Collectors.toList()).subList(0,truncateLen));
            all_segment_ids.add(Arrays.stream(features.get(i).getSegmentIds()).mapToLong(j -> (long)j).boxed().collect(Collectors.toList()).subList(0,truncateLen));
        }

        tensorDatasMap.put("all_label_ids",all_label_ids);
        tensorDatasMap.put("all_seq_lengths",all_seq_lengths);
        tensorDatasMap.put("all_input_ids",all_input_ids);
        tensorDatasMap.put("all_input_mask",all_input_mask);
        tensorDatasMap.put("all_segment_ids",all_segment_ids);
        return tensorDatasMap;
    }


    public static List<int[]> readInputExamples(Context context) {
        AssetManager assetManager = context.getAssets();
        Map<String,List<Integer>> inputDatas = new HashMap<>();
        List<String> inputFileNames = new ArrayList<>();
        inputFileNames.add("attention_mask.txt");
        inputFileNames.add("token_type_ids.txt");
        inputFileNames.add("input_ids.txt");
        for (String fileName : inputFileNames) {
            try (InputStream inputStream = assetManager.open(fileName);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
                String line;
                List<Integer> inputs = new ArrayList<>();
                while ((line = reader.readLine()) != null) {
                    inputs.add(Integer.valueOf(line));
                }
                inputDatas.put(fileName,inputs);
            } catch (IOException e) {
                e.printStackTrace(); // 处理异常，可以根据需要改为抛出异常
            }
        }
        int[] input_ids,token_type_ids, attention_mask;
        attention_mask = inputDatas.get("attention_mask.txt").stream().mapToInt(Integer::intValue).toArray();
        token_type_ids = inputDatas.get("token_type_ids.txt").stream().mapToInt(Integer::intValue).toArray();
        input_ids = inputDatas.get("input_ids.txt").stream().mapToInt(Integer::intValue).toArray();
        List<int[]> returnInputDatas = new ArrayList<>();
        returnInputDatas.add(input_ids);
        returnInputDatas.add(token_type_ids);
        returnInputDatas.add(attention_mask);
        return returnInputDatas;
    }

    public static List<List<int[]>> readMultipleInputExamples(Context context) {
        int batch = 100;
        AssetManager assetManager = context.getAssets();
        Map<String,List<Integer>> inputDatas = new HashMap<>();
        List<List<int[]>> returnDatas = new ArrayList<>();
        List<String> inputFileNames = new ArrayList();
        for (int i = 0; i < batch; i++) {
            List<int[]> InputDatas = new ArrayList<>();
            inputFileNames.clear();
            String input_ids_name = "txtData/"+"input_ids_"+String.valueOf(i)+".txt";
            String input_attention_mask_name = "txtData/"+"attention_mask_"+String.valueOf(i)+".txt";
            inputFileNames.add(input_ids_name);
            inputFileNames.add(input_attention_mask_name);
            for (String fileName : inputFileNames) {
                try (InputStream inputStream = assetManager.open(fileName);
                     BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
                    String line;
                    List<Integer> inputs = new ArrayList<>();
                    while ((line = reader.readLine()) != null) {
                        inputs.add(Integer.valueOf(line));
                    }
                    inputDatas.put(fileName,inputs);
                } catch (IOException e) {
                    e.printStackTrace(); // 处理异常，可以根据需要改为抛出异常
                }
            }
            int[] input_ids, attention_mask;
            attention_mask = inputDatas.get(input_attention_mask_name).stream().mapToInt(Integer::intValue).toArray();
            input_ids = inputDatas.get(input_ids_name).stream().mapToInt(Integer::intValue).toArray();
            InputDatas.add(input_ids);
            InputDatas.add(attention_mask);
            returnDatas.add(InputDatas);
        }
        return returnDatas;
    }

}
