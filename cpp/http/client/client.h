/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <list>

#include "common/client_mgr/client_mgr.h"
#include "utils/threadpool/threadpool.h"

namespace runai::llm::streamer::impl::http
{

namespace api = common::backend_api;
using common::ResponseCode;

class Client : public common::IClient
{
 public:
    explicit Client(const api::ObjectClientConfig_t & config);
    ~Client() override;

    bool verify_credentials(const api::ObjectClientConfig_t & config) const override;
    ResponseCode read(const char * url, api::ObjectRange_t range, char * buffer, api::ObjectRequestId_t id);
    ResponseCode wait(api::ObjectCompletionEvent_t * events, unsigned capacity, unsigned & count,
                      api::ObjectWaitMode_t mode);
    void stop();
    void release();

 private:
    using Events = std::list<api::ObjectCompletionEvent_t>;
    struct Request
    {
        std::string url;
        api::ObjectRange_t range;
        char * buffer;
        Events::iterator event;
    };
    void perform(Request && request);

    unsigned _readers;
    std::atomic<bool> _stopped{false};
    std::mutex _mutex;
    std::condition_variable _changed;
    unsigned _waiters = 0;
    Events _pending;
    Events _ready;
    utils::ThreadPool<Request> _pool;
};

} // namespace runai::llm::streamer::impl::http
