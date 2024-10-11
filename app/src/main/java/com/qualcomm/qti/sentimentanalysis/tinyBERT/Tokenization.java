package com.qualcomm.qti.sentimentanalysis.tinyBERT;

import static com.qualcomm.qti.sentimentanalysis.tokenization.BasicTokenizer.whitespaceTokenize;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.*;
import android.util.Log;
import android.content.Context;

import java.io.InputStream;
import java.io.InputStreamReader;

public class Tokenization {

    public static List<String> whitespace_tokenize(String text) {
        /** Runs basic whitespace cleaning and splitting on a piece of text. */
        text = text.trim();
        if (text.isEmpty()) {
            return new ArrayList<>();
        }
        String[] tokens = text.split("\\s+");
        List<String> tokenList = new ArrayList<>();
        for (String token : tokens) {
            tokenList.add(token);
        }
        return tokenList;
    }


    public static boolean isControl(char c) {
        // Check if the character is a tab, newline, or carriage return
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }

        // Get the character type using the Character class
        int charType = Character.getType(c);

        // Check if the character type is a control character
        if (charType == Character.CONTROL || charType == Character.FORMAT) {
            return true;
        }

        return false;
    }


    public static boolean isPunctuation(char c) {
        int cp = (int) c;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }
        int charType = Character.getType(c);
        if (charType == Character.DASH_PUNCTUATION || charType == Character.START_PUNCTUATION || charType == Character.END_PUNCTUATION || charType == Character.CONNECTOR_PUNCTUATION || charType == Character.OTHER_PUNCTUATION || charType == Character.INITIAL_QUOTE_PUNCTUATION || charType == Character.FINAL_QUOTE_PUNCTUATION) {
            return true;
        }

        return false;
    }

    public static boolean isWhitespace(char c) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return true;
        }
        int charType = Character.getType(c);
        if (charType == Character.SPACE_SEPARATOR) {
            return true;
        }
        return false;
    }


    public class BasicTokenizer {

        private final boolean doLowerCase;
        private final List<String> neverSplit;

        public BasicTokenizer(boolean doLowerCase, List<String> neverSplit) {
            this.doLowerCase = doLowerCase;
            this.neverSplit = neverSplit;
        }

        public List<String> tokenize(String text) {
            text = cleanText(text);
            text = tokenizeChineseChars(text);
            List<String> origTokens = whitespaceTokenize(text);
            List<String> splitTokens = new ArrayList<>();
            for (String token : origTokens) {
                if (doLowerCase && !neverSplit.contains(token)) {
                    token = token.toLowerCase();
                    token = runStripAccents(token);
                }
                splitTokens.addAll(runSplitOnPunc(token));
            }

            List<String> outputTokens = whitespaceTokenize(String.join(" ", splitTokens));
            return outputTokens;
        }

        private String runStripAccents(String text) {
            text = Normalizer.normalize(text, Normalizer.Form.NFD);
            StringBuilder output = new StringBuilder();
            for (char c : text.toCharArray()) {
                int charType = Character.getType(c);
                if (charType != Character.NON_SPACING_MARK) {
                    output.append(c);
                }
            }
            return output.toString();
        }

        private List<String> runSplitOnPunc(String text) {
            if (neverSplit.contains(text)) {
                return Arrays.asList(text);
            }
            List<String> output = new ArrayList<>();
            StringBuilder buffer = new StringBuilder();
            for (char c : text.toCharArray()) {
                if (isPunctuation(c)) {
                    if (buffer.length() > 0) {
                        output.add(buffer.toString());
                        buffer.setLength(0);
                    }
                    output.add(Character.toString(c));
                } else {
                    buffer.append(c);
                }
            }
            if (buffer.length() > 0) {
                output.add(buffer.toString());
            }
            return output;
        }

        private String tokenizeChineseChars(String text) {
            StringBuilder output = new StringBuilder();
            for (char c : text.toCharArray()) {
                int codePoint = (int) c;
                if (isChineseChar(codePoint)) {
                    output.append(' ');
                    output.append(c);
                    output.append(' ');
                } else {
                    output.append(c);
                }
            }
            return output.toString();
        }

        private boolean isChineseChar(int codePoint) {
            return (codePoint >= 0x4E00 && codePoint <= 0x9FFF) || (codePoint >= 0x3400 && codePoint <= 0x4DBF) ||
                    (codePoint >= 0x20000 && codePoint <= 0x2A6DF) || (codePoint >= 0x2A700 && codePoint <= 0x2B73F) ||
                    (codePoint >= 0x2B740 && codePoint <= 0x2B81F) || (codePoint >= 0x2B820 && codePoint <= 0x2CEAF) ||
                    (codePoint >= 0xF900 && codePoint <= 0xFAFF) || (codePoint >= 0x2F800 && codePoint <= 0x2FA1F);
        }

        private String cleanText(String text) {
            StringBuilder output = new StringBuilder();
            for (char c : text.toCharArray()) {
                int codePoint = (int) c;
                if (codePoint == 0 || codePoint == 0xFFFD || isControl(c)) {
                    continue;
                }
                if (isWhitespace(c)) {
                    output.append(' ');
                } else {
                    output.append(c);
                }
            }
            return output.toString();
        }

    }


    public class BertTokenizer {
        private  Map<Integer, String> idsToTokens;
        private  boolean doBasicTokenize;
        private  BasicTokenizer basicTokenizer;
        private  WordpieceTokenizer wordpieceTokenizer;
        private  int maxLen;
        private  boolean basicOnly;
        private  Context context;

        public BertTokenizer(Context context) {
            this.context = context;
        }

        public BertTokenizer(Context context, Map<String, Integer> vocab,boolean doLowerCase, Integer maxLen, boolean doBasicTokenize, boolean basicOnly,
                             List<String> neverSplit) throws IOException {
            this.context = context;
            this.idsToTokens = new LinkedHashMap<>();
            for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
                this.idsToTokens.put(entry.getValue(), entry.getKey());
            }
            this.doBasicTokenize = doBasicTokenize;
            if (doBasicTokenize) {
                this.basicTokenizer = new BasicTokenizer(doLowerCase, neverSplit);
            } else {
                this.basicTokenizer = null;
            }
            this.wordpieceTokenizer = new WordpieceTokenizer(vocab,"[UNK]",100);
            this.maxLen = maxLen != null ? maxLen : Integer.MAX_VALUE;
            this.basicOnly = basicOnly;
        }

        public List<String> tokenize(String text) {
            List<String> splitTokens = new ArrayList<>();
            if (doBasicTokenize) {
                for (String token : basicTokenizer.tokenize(text)) {
                    if (basicOnly) {
                        splitTokens.add(token);
                    } else {
                        for (String subToken : wordpieceTokenizer.tokenize(token)) {
                            splitTokens.add(subToken);
                        }
                    }
                }
            } else {
                splitTokens = wordpieceTokenizer.tokenize(text);
            }
            return splitTokens;
        }

        public List<Integer> convertTokensToIds(Map<String,Integer> vocab,List<String> tokens) {
            List<Integer> ids = new ArrayList<>();
            for (String token : tokens) {
                ids.add(vocab.getOrDefault(token, vocab.get("[UNK]")));
            }
            if (ids.size() > maxLen) {
                Log.i("BERT_TOKENIZER_TAG",String.format(
                        "Token indices sequence length is longer than the specified maximum "
                                + "sequence length for this BERT model (%d > %d). Running this "
                                + "sequence through BERT will result in indexing errors",
                        ids.size(), maxLen));
            }
            return ids;
        }


        public List<String> convertIdsToTokens(List<Integer> ids) {
            List<String> tokens = new ArrayList<>();
            for (Integer id : ids) {
                tokens.add(idsToTokens.get(id));
            }
            return tokens;
        }



    }

    public static Map<String, Integer> readVocabFile(String fileName,Context context) {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        int index = 0;
        try (InputStream inputStream = context.getAssets().open(fileName);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"))) {
            String token;
            while ((token = reader.readLine()) != null) {
                token = token.trim();
                if (!token.isEmpty()) {
                    vocab.put(token, index);
                    index++;
                }
            }
        } catch (IOException e) {
            e.printStackTrace(); // 处理异常
        }
        return vocab;
    }

    public BertTokenizer getBertTokenizer(Context context) throws IOException {
        String resolvedVocabFile = "vocab.txt";
        Map<String, Integer> vocab = readVocabFile(resolvedVocabFile,context);
        List<String> neverSplit = Arrays.asList("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]");
        int maxLen = 512;
        BertTokenizer bertTokenizer = new BertTokenizer(context,vocab,true,maxLen,true,false,neverSplit);
        return bertTokenizer;
    }


    public class WordpieceTokenizer {

        private final Map<String, Integer> vocab;
        private final String unkToken;
        private final int maxInputCharsPerWord;

        public WordpieceTokenizer(Map<String, Integer> vocab, String unkToken, int maxInputCharsPerWord) {
            this.vocab = vocab;
            this.unkToken = unkToken;
            this.maxInputCharsPerWord = maxInputCharsPerWord;
        }

        public List<String> tokenize(String text) {
            List<String> outputTokens = new ArrayList<>();
            for (String token : whitespaceTokenize(text)) {
                List<String> chars = Arrays.asList(token.split(""));
                if (chars.size() > maxInputCharsPerWord) {
                    outputTokens.add(unkToken);
                    continue;
                }

                boolean isBad = false;
                int start = 0;
                List<String> subTokens = new ArrayList<>();
                while (start < chars.size()) {
                    int end = chars.size();
                    String curSubstr = null;
                    while (start < end) {
                        String substr = String.join("", chars.subList(start, end));
                        if (start > 0) {
                            substr = "##" + substr;
                        }
                        if (vocab.containsKey(substr)) {
                            curSubstr = substr;
                            break;
                        }
                        end--;
                    }
                    if (curSubstr == null) {
                        isBad = true;
                        break;
                    }
                    subTokens.add(curSubstr);
                    start = end;
                }

                if (isBad) {
                    outputTokens.add(unkToken);
                } else {
                    outputTokens.addAll(subTokens);
                }
            }
            return outputTokens;
        }

    }



}
