/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http/client/range_get.h"

#include <curl/curl.h>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>
#include <strings.h>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl::http
{
namespace
{

bool equal_ci(std::string_view a, std::string_view b)
{
    return a.size() == b.size() && strncasecmp(a.data(), b.data(), a.size()) == 0;
}

bool take_number(std::string_view & value, size_t & number)
{
    auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{}) return false;
    value.remove_prefix(result.ptr - value.data());
    return true;
}

bool take(std::string_view & value, char separator)
{
    if (value.empty() || value.front() != separator) return false;
    value.remove_prefix(1);
    return true;
}

bool matches_range(std::string_view value, size_t first, size_t last)
{
    if (!equal_ci(value.substr(0, 6), "bytes ")) return false;
    value.remove_prefix(6);
    size_t start, end, total;
    return take_number(value, start) && start == first && take(value, '-') &&
           take_number(value, end) && end == last && take(value, '/') &&
           (value == "*" || (take_number(value, total) && value.empty() && total > end));
}

struct Transfer
{
    common::backend_api::ObjectRange_t range;
    char * buffer;
    size_t & written;
    const std::atomic<bool> & stopped;

    long status = 0;
    bool valid_range = false;
    bool headers_done = false;
    bool oversized = false;
};

size_t header(char * data, size_t, size_t bytes, void * context)
{
    auto & t = *static_cast<Transfer *>(context);
    std::string_view line(data, bytes);
    line = line.substr(0, line.find_last_not_of("\r\n \t") + 1);
    if (line.substr(0, 5) == "HTTP/")
    {
        const auto space = line.find(' ');
        if (space == line.npos) return 0;
        line.remove_prefix(space + 1);
        if (std::from_chars(line.data(), line.data() + line.size(), t.status).ec != std::errc{}) return 0;
        t.valid_range = false;
        t.headers_done = false;
    }
    else if (!t.headers_done)
    {
        if (line.empty())
        {
            t.headers_done = true;
            // Allow interim responses and redirects. Reject 200 before consuming its body.
            if (t.status >= 200 && (t.status < 300 || t.status >= 400) &&
                !(t.status == 206 && t.valid_range)) return 0;
        }
        else if (t.status == 206)
        {
            const auto colon = line.find(':');
            if (colon == line.npos) return 0;
            const auto field = line.substr(0, colon);
            line.remove_prefix(colon + 1);
            while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.remove_prefix(1);
            if (equal_ci(field, "Content-Range"))
            {
                if (t.valid_range || !matches_range(line, t.range.offset, t.range.offset + (t.range.length - 1)))
                    return 0;
                t.valid_range = true;
            }
            if (equal_ci(field, "Content-Encoding") && !equal_ci(line, "identity")) return 0;
        }
    }
    return bytes;
}

size_t body(char * data, size_t, size_t bytes, void * context)
{
    auto & t = *static_cast<Transfer *>(context);
    if (t.stopped) return 0;
    if (t.status >= 300 && t.status < 400) return bytes;
    if (t.status != 206 || !t.headers_done || !t.valid_range) return 0;
    if (bytes > t.range.length - t.written)
    {
        t.oversized = true;
        return 0;
    }
    std::memcpy(t.buffer + t.written, data, bytes);
    t.written += bytes;
    return bytes;
}

int progress(void * context, curl_off_t, curl_off_t, curl_off_t, curl_off_t)
{
    return static_cast<Transfer *>(context)->stopped.load() ? 1 : 0;
}

} // namespace

common::ResponseCode range_get(const std::string & url, common::backend_api::ObjectRange_t range,
                               char * buffer, size_t & written, const std::atomic<bool> & stopped)
{
    using common::ResponseCode;
    if (range.length == 0) return ResponseCode::Success;
    const std::string bounds = std::to_string(range.offset) + "-" + std::to_string(range.offset + (range.length - 1));
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
    if (!curl) return ResponseCode::UnknownError;
    Transfer transfer{range, buffer, written, stopped};

    const auto set = [&](CURLoption option, auto value) { return curl_easy_setopt(curl.get(), option, value) == CURLE_OK; };
    if (!set(CURLOPT_URL, url.c_str()) || !set(CURLOPT_RANGE, bounds.c_str()) ||
        !set(CURLOPT_PATH_AS_IS, 1L) || !set(CURLOPT_FOLLOWLOCATION, 1L) || !set(CURLOPT_MAXREDIRS, 10L) ||
        !set(CURLOPT_PROTOCOLS, long(CURLPROTO_HTTP | CURLPROTO_HTTPS)) ||
        !set(CURLOPT_REDIR_PROTOCOLS, long(CURLPROTO_HTTP | CURLPROTO_HTTPS)) ||
        !set(CURLOPT_DISALLOW_USERNAME_IN_URL, 1L) || !set(CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L) ||
        !set(CURLOPT_NOSIGNAL, 1L) || !set(CURLOPT_NOPROGRESS, 0L) ||
        !set(CURLOPT_XFERINFOFUNCTION, progress) || !set(CURLOPT_XFERINFODATA, &transfer) ||
        !set(CURLOPT_HEADERFUNCTION, header) || !set(CURLOPT_HEADERDATA, &transfer) ||
        !set(CURLOPT_WRITEFUNCTION, body) || !set(CURLOPT_WRITEDATA, &transfer))
    {
        return ResponseCode::UnknownError;
    }
    if (const auto * ca = std::getenv("CURL_CA_BUNDLE"); ca && *ca && !set(CURLOPT_CAINFO, ca))
    {
        return ResponseCode::CaFileNotFound;
    }
    const auto result = curl_easy_perform(curl.get());
    if (result == CURLE_OK && transfer.status == 206 && transfer.valid_range && written == range.length)
        return ResponseCode::Success;
    LOG(ERROR) << "HTTP range GET failed (status " << transfer.status << ", " << curl_easy_strerror(result)
               << ", wrote " << written << " of " << range.length << " bytes)";
    if (transfer.valid_range && (transfer.oversized || result == CURLE_PARTIAL_FILE || result == CURLE_OK))
        return ResponseCode::FileTruncatedError;
    if (!stopped && (transfer.status == 200 || (transfer.status == 206 && result == CURLE_WRITE_ERROR)))
        return ResponseCode::HTTPInvalidRangeResponse;
    return ResponseCode::FileAccessError;
}

} // namespace runai::llm::streamer::impl::http
