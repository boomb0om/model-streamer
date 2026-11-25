/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http/http.h"
#include "http/client/client.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

#include <curl/curl.h>

namespace runai::llm::streamer::impl::http
{

inline constexpr char ClientName[] = "HTTP";
using Clients = common::ClientMgr<Client, ClientName>;

namespace
{
struct Backend
{
    // The ABI exposes one window per backend. A smaller client must not inherit a larger client's window.
    std::atomic<size_t> max_inflight_bytes{std::numeric_limits<size_t>::max()};
};
} // namespace

// Keep exceptions from allocations and thread creation inside the plugin ABI.
template<typename F>
static ResponseCode guarded(F && call)
{
    try
    {
        return call();
    }
    catch (const std::exception & e)
    {
        LOG(ERROR) << "HTTP backend error: " << e.what();
        return ResponseCode::UnknownError;
    }
}

ResponseCode obj_open_backend(api::ObjectBackendHandle_t * handle)
{
    if (!handle) return ResponseCode::InvalidParameterError;
    *handle = nullptr;
    return guarded([&] {
        auto backend = std::make_unique<Backend>();
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) return ResponseCode::UnknownError;
        *handle = backend.release();
        return ResponseCode::Success;
    });
}

ResponseCode obj_close_backend(api::ObjectBackendHandle_t handle)
{
    delete static_cast<Backend *>(handle);
    curl_global_cleanup();
    return ResponseCode::Success;
}

api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy()
{
    return api::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN;
}

ResponseCode obj_get_backend_config(api::ObjectBackendHandle_t handle, const char * key,
                                    char * value, unsigned * size)
{
    if (!handle || !key || !value || !size || std::strcmp(key, "max_inflight_bytes") != 0)
        return ResponseCode::InvalidParameterError;

    const auto window = static_cast<Backend *>(handle)->max_inflight_bytes.load();
    if (window == std::numeric_limits<size_t>::max()) return ResponseCode::InvalidParameterError;

    char buffer[std::numeric_limits<size_t>::digits10 + 2];
    const int length = std::snprintf(buffer, sizeof(buffer), "%zu", window);
    if (length < 0 || static_cast<size_t>(length) >= sizeof(buffer)) return ResponseCode::UnknownError;
    const auto needed = static_cast<unsigned>(length) + 1;
    if (*size < needed)
    {
        *size = needed;
        return ResponseCode::InvalidParameterError;
    }
    std::memcpy(value, buffer, needed);
    *size = needed - 1;
    return ResponseCode::Success;
}

ResponseCode obj_create_client(api::ObjectBackendHandle_t backend, const api::ObjectClientConfig_t * config,
                               api::ObjectClientHandle_t * handle)
{
    if (!backend || !config || !handle) return ResponseCode::InvalidParameterError;
    *handle = nullptr;
    return guarded([&] {
        auto * client = Clients::pop(*config);
        auto & window = static_cast<Backend *>(backend)->max_inflight_bytes;
        const auto capacity = client->max_inflight_bytes();
        auto previous = window.load();
        while (capacity < previous && !window.compare_exchange_weak(previous, capacity)) {}
        *handle = client;
        return ResponseCode::Success;
    });
}

ResponseCode obj_remove_client(api::ObjectClientHandle_t handle)
{
    return guarded([&] {
        if (auto * client = static_cast<Client *>(handle))
        {
            client->release();
            Clients::push(client);
        }
        return ResponseCode::Success;
    });
}

ResponseCode obj_cancel_all_reads()
{
    return guarded([] { Clients::stop(); return ResponseCode::Success; });
}

ResponseCode obj_remove_all_clients()
{
    return guarded([] { Clients::clear(); return ResponseCode::Success; });
}

ResponseCode obj_request_read(api::ObjectClientHandle_t handle, const char * path, api::ObjectRange_t range,
                              char * buffer, api::ObjectRequestId_t id)
{
    if (!handle) return ResponseCode::InvalidParameterError;
    return guarded([&] { return static_cast<Client *>(handle)->read(path, range, buffer, id); });
}

ResponseCode obj_wait_for_completions(api::ObjectClientHandle_t handle, api::ObjectCompletionEvent_t * events,
                                      unsigned capacity, unsigned * count, api::ObjectWaitMode_t mode)
{
    if (!handle || !events || !capacity || !count) return ResponseCode::InvalidParameterError;
    return guarded([&] { return static_cast<Client *>(handle)->wait(events, capacity, *count, mode); });
}

ResponseCode obj_list_files(api::ObjectClientHandle_t, const char *, int, api::ObjectFileEntry_t **, unsigned *)
{
    return ResponseCode::HTTPListingNotSupported;
}

void obj_free_file_list(api::ObjectFileEntry_t *, unsigned)
{}

} // namespace runai::llm::streamer::impl::http
