/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace runai::llm::streamer::common::s3
{

// For diagnostics only. Requests always use the original URI.
std::string_view display_path(std::string_view path);

struct StorageUri
{
    StorageUri(const std::string & uri);

    std::string scheme;
    std::string uri;
    std::string bucket;
    std::string path;

    bool is_http() const;
    bool is_gcs() const;
    bool is_azure() const;
};

struct StorageUri_C
{
    StorageUri_C(const StorageUri & uri);

    const char * bucket;
    const char * path;
};

std::ostream & operator<<(std::ostream &, const StorageUri &);

}; //namespace runai::llm::streamer::common::s3
