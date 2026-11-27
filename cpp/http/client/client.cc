/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 boomb0om
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http/client/client.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>

#include "http/client/range_get.h"
#include "utils/env/env.h"

namespace runai::llm::streamer::impl::http
{

namespace
{

constexpr char ca_bundle_env[] = "RUNAI_STREAMER_HTTP_CA_BUNDLE";

size_t inflight_window(unsigned threads, size_t chunk_bytesize)
{
    const auto chunks = std::max(2.0, api::inflight_window_margin() * threads);
    const auto bytes = chunks * chunk_bytesize;
    constexpr auto unbounded = std::numeric_limits<size_t>::max();
    return bytes < static_cast<double>(unbounded) ? static_cast<size_t>(bytes) : unbounded - 1;
}

} // namespace

class Client::Worker final : public utils::Worker<Request>
{
 public:
    explicit Worker(Client & client) : _client(client) {}

    void execute(Request && request, std::atomic<bool> &) override
    {
        _client.perform(_curl.get(), std::move(request));
    }

    void drain(std::atomic<bool> &) override {}
    bool idle() const override { return true; }

 private:
    Client & _client;
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> _curl{curl_easy_init(), curl_easy_cleanup};
};

Client::Client(const api::ObjectClientConfig_t & config) :
    _readers(config.concurrent_readers),
    _threads(std::max(1U, std::max(8U, std::thread::hardware_concurrency()) / std::max(1U, _readers))),
    _chunk_bytesize(std::max(size_t{1}, config.default_storage_chunk_size)),
    _max_inflight_bytes(inflight_window(_threads, _chunk_bytesize)),
    _ca_bundle(utils::getenv<std::string>(ca_bundle_env, "")),
    _pool([this] { return std::make_unique<Worker>(*this); }, _threads)
{}

Client::~Client()
{
    stop();
}

bool Client::verify_credentials(const api::ObjectClientConfig_t & config) const
{
    return !_stopped && config.concurrent_readers == _readers &&
           std::max(size_t{1}, config.default_storage_chunk_size) == _chunk_bytesize &&
           utils::getenv<std::string>(ca_bundle_env, "") == _ca_bundle;
}

ResponseCode Client::read(const char * url, api::ObjectRange_t range, char * buffer, api::ObjectRequestId_t id)
{
    if (!url || (range.length && (!buffer || range.length - 1 > std::numeric_limits<size_t>::max() - range.offset)))
    {
        return ResponseCode::InvalidParameterError;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_stopped)
    {
        return ResponseCode::FinishedError;
    }
    auto event = _pending.insert(_pending.end(), {id, ResponseCode::FileAccessError, 0});
    try
    {
        _pool.push({url, range, buffer, event});
    }
    catch (...)
    {
        _pending.erase(event);
        throw;
    }
    return ResponseCode::Success;
}

void Client::perform(CURL * curl, Request && request)
{
    auto & event = *request.event;
    try
    {
        if (!_stopped)
        {
            event.response_code = range_get(curl, request.url, request.range, request.buffer,
                                            event.bytes_transferred, _stopped, _ca_bundle);
        }
    }
    catch (const std::exception &)
    {
        event.response_code = ResponseCode::UnknownError;
    }

    // Publish the preallocated event only after the last possible write to the destination.
    std::lock_guard<std::mutex> lock(_mutex);
    _ready.splice(_ready.end(), _pending, request.event);
    _changed.notify_all();
}

ResponseCode Client::wait(api::ObjectCompletionEvent_t * events, unsigned capacity, unsigned & count,
                          api::ObjectWaitMode_t mode)
{
    std::unique_lock<std::mutex> lock(_mutex);
    ++_waiters;
    auto available = [this] { return !_ready.empty() || _pending.empty(); };
    if (mode == api::OBJECT_WAIT_MODE_BLOCK)
    {
        _changed.wait(lock, available);
    }
    else if (mode == api::OBJECT_WAIT_MODE_TIMED_BLOCK)
    {
        // The current backend ABI has no timeout argument.
        _changed.wait_for(lock, std::chrono::milliseconds(100), available);
    }
    count = 0;
    while (count < capacity && !_ready.empty())
    {
        events[count++] = _ready.front();
        _ready.pop_front();
    }
    if (--_waiters == 0 && _stopped) _changed.notify_all();
    return count == 0 && _pending.empty() && mode != api::OBJECT_WAIT_MODE_NON_BLOCKING
        ? ResponseCode::FinishedError : ResponseCode::Success;
}

void Client::stop()
{
    std::unique_lock<std::mutex> lock(_mutex);
    _stopped = true;
    _changed.wait(lock, [this] { return _pending.empty() && _waiters == 0; });
}

void Client::release()
{
    std::unique_lock<std::mutex> lock(_mutex);
    // A waiter already inside the ABI must collect its event before the client can be reused.
    if (!_pending.empty() || _waiters != 0)
    {
        _stopped = true;
        _changed.wait(lock, [this] { return _pending.empty() && _waiters == 0; });
    }
    _ready.clear();
}

} // namespace runai::llm::streamer::impl::http
