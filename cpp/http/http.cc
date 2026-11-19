/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http/http.h"
#include "http/client/client.h"

#include <curl/curl.h>

namespace runai::llm::streamer::impl::http
{

inline constexpr char ClientName[] = "HTTP";
using Clients = common::ClientMgr<Client, ClientName>;

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
    return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK ? ResponseCode::Success : ResponseCode::UnknownError;
}

ResponseCode obj_close_backend(api::ObjectBackendHandle_t)
{
    curl_global_cleanup();
    return ResponseCode::Success;
}

api::ObjectShutdownPolicy_t obj_get_backend_shutdown_policy()
{
    return api::OBJECT_SHUTDOWN_POLICY_ON_STREAMER_SHUTDOWN;
}

ResponseCode obj_get_backend_config(api::ObjectBackendHandle_t, const char *, char *, unsigned *)
{
    return ResponseCode::InvalidParameterError;
}

ResponseCode obj_create_client(api::ObjectBackendHandle_t, const api::ObjectClientConfig_t * config,
                               api::ObjectClientHandle_t * handle)
{
    if (!config || !handle) return ResponseCode::InvalidParameterError;
    *handle = nullptr;
    return guarded([&] {
        *handle = Clients::pop(*config);
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
