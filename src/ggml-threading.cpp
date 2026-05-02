#include "ggml-threading.h"  // 引入 ggml-threading.h 头文件
#include <mutex>  // 引入 mutex 头文件

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}
