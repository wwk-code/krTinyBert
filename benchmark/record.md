# SentimentAnalysis Benchmark(NPU vs CPU)


## 1. Infer result comparasion

### 1.1. text: "I am afraid I can't do that"

| Target     | NPU | CPU   |
| ---------- | --- | ----- |
| Positivity | 2%  | 2%    |
| Negativity | 98% | 98%   |
| Infer Time | 7ms | 151ms |

### 1.2. text: "I am glad to tell you that we can do it together"

| Target     | NPU | CPU   |
| :--------- | --- | ----- |
| Positivity | 99% | 99%   |
| Negativity | 1%  | 1%    |
| Infer Time | 7ms | 152ms |

### 1.3. text: "Too long to wait"

| Target     | NPU  | CPU   |
| ---------- | ---- | ----- |
| Positivity | 1%   | 1%    |
| Negativity | 99%  | 99%   |
| Infer Time | 8 ms | 162ms |



## 2. Device resources consumption comparasion

**Text:  "I'm afraid I can't do that"**

| Target       | NPU    | CPU   |
| ------------ | ------ | ----- |
| CPU Usage    | 5%     | 56%   |
| Memory Usage | 560 MB | 575MB |
