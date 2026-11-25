/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <string>

#include <curl/curl.h>

#include "common/backend_api/object_storage/object_storage.h"

namespace runai::llm::streamer::impl::http
{

common::ResponseCode range_get(CURL * curl, const std::string & url, common::backend_api::ObjectRange_t range,
                               char * buffer, size_t & written, const std::atomic<bool> & stopped,
                               const std::string & ca_bundle, long stall_timeout_seconds = 30);

} // namespace runai::llm::streamer::impl::http
