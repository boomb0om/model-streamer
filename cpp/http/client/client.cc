/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http/client/client.h"

#include <algorithm>
#include <chrono>
#include <limits>

#include "http/client/range_get.h"

namespace runai::llm::streamer::impl::http
{

Client::Client(const api::ObjectClientConfig_t & config) :
    _readers(config.concurrent_readers),
    _pool([this](Request && request, std::atomic<bool> &) { perform(std::move(request)); },
          std::max(1U, std::max(8U, std::thread::hardware_concurrency()) / std::max(1U, _readers)))
{}

Client::~Client()
{
    stop();
}

bool Client::verify_credentials(const api::ObjectClientConfig_t & config) const
{
    return !_stopped && config.concurrent_readers == _readers;
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

void Client::perform(Request && request)
{
    auto & event = *request.event;
    try
    {
        if (!_stopped)
        {
            event.response_code = range_get(request.url, request.range, request.buffer, event.bytes_transferred, _stopped);
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
