#pragma once  // 防止重复包含

#include <cstddef>  // 引入 cstddef 头文件
#include <cstdint>  // 引入 cstdint 头文件
#include <memory>  // 引入 memory 头文件

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;  // 类型定义

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {  // 结构体定义
    ~socket_t();

    bool send_data(const void * data, size_t size);  // send_data
    bool recv_data(void * data, size_t size);  // recv_data

    socket_ptr accept();  // accept

    void get_caps(uint8_t * local_caps);  // get_caps
    void update_caps(const uint8_t * remote_caps);  // update_caps

    static socket_ptr create_server(const char * host, int port);  // create_server
    static socket_ptr connect(const char * host, int port);  // connect

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);  // socket_t
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();  // rpc_transport_init
void rpc_transport_shutdown();  // rpc_transport_shutdown
