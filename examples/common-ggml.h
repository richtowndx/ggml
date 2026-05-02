#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件

#include <fstream>  // 引入 fstream 头文件
#include <vector>  // 引入 vector 头文件
#include <string>  // 引入 string 头文件

enum ggml_ftype ggml_parse_ftype(const char * str);  // ggml_parse_ftype

void ggml_print_ftypes(FILE * fp = stderr);  // ggml_print_ftypes

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);
