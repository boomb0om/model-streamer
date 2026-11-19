/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex>
#include <vector>

#include "common/storage_uri/storage_uri.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::common::s3
{

static const std::string gcsProtocol("gs");
static const std::string azureProtocol("az");

static bool http_url(std::string_view path)
{
    return path.substr(0, 7) == "http://" || path.substr(0, 8) == "https://";
}

std::string_view display_path(std::string_view path)
{
    return http_url(path) ? path.substr(0, path.find_first_of("?#")) : path;
}

StorageUri::StorageUri(const std::string & uri) : uri(uri)
{
    if (http_url(uri))
    {
        // Preserve the full URL in uri for libcurl. Keep the signature out of the parsed
        // fields, which are also used for diagnostics. Let libcurl validate the authority.
        const auto name = display_path(uri);
        const auto colon = name.find(':');
        const auto slash = name.find('/', colon + 3);
        scheme = name.substr(0, colon);
        bucket = name.substr(colon + 3, slash - (colon + 3));
        if (slash != name.npos) path = name.substr(slash + 1);
        return;
    }
    static const std::regex awsRegex("^(s3|gs|az)://([^/]+)/(.+)$");

    std::smatch match;

    LOG(SPAM) << "Checking: " << uri;

    if (!std::regex_match(uri, match, awsRegex))
    {
        LOG(SPAM) << "'" << uri << "' is not in s3 format";
        throw std::exception();
    }

    scheme = match[1];
    bucket = match[2];
    path = match[3];

    LOG(SPAM) << "scheme: " << scheme << " bucket: " << bucket << " path: " << path;
}

std::ostream & operator<<(std::ostream & os, const StorageUri & uri)
{
    return os << "scheme: " << uri.scheme << " bucket: " << uri.bucket << " path: " << uri.path;
}

bool StorageUri::is_http() const
{
    return scheme == "http" || scheme == "https";
}

bool StorageUri::is_gcs() const
{
    return scheme == gcsProtocol;
}

bool StorageUri::is_azure() const
{
    return scheme == azureProtocol;
}

StorageUri_C::StorageUri_C(const StorageUri & uri) :
    bucket(uri.bucket.c_str()),
    path(uri.path.c_str())
{}

}; // namespace runai::llm::streamer::common::s3
