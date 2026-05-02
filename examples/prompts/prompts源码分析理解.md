# Prompts 源码分析理解

## 概述

prompts 目录不是源码，而是一个**测试数据集集合**，为 GGML 示例程序（主要是 GPT-2、GPT-J、GPT-NeoX 等语言模型）提供标准化的输入文本和预期的 tokenization 结果。这些文件用于验证 tokenizer 的正确性，确保不同实现之间的兼容性。

## 文件结构

| 文件 | 对应模型/用途 |
|------|--------------|
| `gpt-2.txt` | GPT-2 tokenizer 测试用例 |
| `gpt-j.txt` | GPT-J tokenizer 测试用例 |
| `gpt-neox.txt` | GPT-NeoX tokenizer 测试用例 |
| `gpt-neox-japanese.txt` | GPT-NeoX 日语 tokenizer 测试用例 |
| `dolly-v2.txt` | Dolly v2 模型 tokenizer 测试用例 |
| `replit.txt` | Replit 模型 tokenizer 测试用例 |
| `starcoder.txt` | StarCoder 模型 tokenizer 测试用例 |
| `whisper.txt` | Whisper 语音模型 tokenizer 测试用例 |
| `polyglot-ko.txt` | Polyglot 韩语模型 tokenizer 测试用例 |
| `gpt-2-chinese.txt` | GPT-2 中文 tokenizer 测试用例 |
| `test-cases.txt` | 通用多语言测试用例 |
| `tokenize_huggingface.py` | HuggingFace tokenizer 对比验证脚本 |

## 文件格式分析

### GPT-2 格式（`gpt-2.txt`）

每行格式：`文本 => token1,token2,...`

```
Hello World! => 15496,2159,0
I can't believe it's already Friday!" => 40,460,470,1975,340,338,1541,3217,2474
```

这表示 `Hello World!` 应被 tokenize 为 `[15496, 2159, 0]` 三个 token。

### 测试用例类型

从 `test-cases.txt` 可以看出，测试用例覆盖了多种语言和特殊场景：

1. **多语言**：English、中文、日语、韩语
2. **特殊字符**：URL（`https://www.example.com`）、邮件地址、温度符号（°C）
3. **混合文本**：leet speak（`Th@nk y0u`）、混合大小写
4. **标点和引号**：嵌套引号、撇号
5. **数字和货币**：`$19.99`、`25.5°C`、时间 `2:30 p.m.`

### Python 验证脚本 (`tokenize_huggingface.py`)

用于将 HuggingFace tokenizer 的输出转换为 GGML 测试格式，作为参考标准。

## 设计目的

1. **Tokenizer 正确性验证**：确保 GGML 实现的 tokenizer 与原始模型（HuggingFace）的 tokenization 结果一致
2. **回归测试**：当修改 tokenizer 代码时，可以快速验证是否引入了 bug
3. **多模型兼容性**：不同模型使用不同的 tokenizer（如 GPT-2 的 BPE、GPT-NeoX 的 sentencepiece），测试文件分别覆盖
4. **边界条件覆盖**：包含各种边界情况，如空格、特殊字符、混合语言

## 展示的 GGML 关键特性

1. **多 Tokenizer 支持**：GGML 需要支持各种模型的 tokenizer（BPE、sentencepiece、Unigram 等）
2. **跨语言处理**：中文、日语、韩语等 CJK 字符的正确 tokenization
3. **与 HuggingFace 的兼容性**：确保 GGML 的 tokenization 结果与主流框架一致
