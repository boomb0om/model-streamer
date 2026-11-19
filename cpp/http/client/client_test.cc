/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http/client/client.h"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <thread>

#include <curl/curl.h>
#include <gtest/gtest.h>

namespace runai::llm::streamer::impl::http
{
namespace
{
using namespace std::chrono_literals;

// Each loopback listener serves one request. Tests can hold the last part of the body until
// cancellation has completed, independently of the order in which workers connect.
class Server
{
 public:
    Server(std::string reply, std::string tail = "")
    {
        _socket = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t size = sizeof(address);
        if (_socket < 0 || bind(_socket, reinterpret_cast<sockaddr *>(&address), size) ||
            listen(_socket, 1) || getsockname(_socket, reinterpret_cast<sockaddr *>(&address), &size))
            throw std::runtime_error("cannot listen on loopback");
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)) + "/model%2F1?b=2&a=1";
        _thread = std::thread([this, reply, tail] {
            pollfd pfd{_socket, POLLIN, 0};
            if (poll(&pfd, 1, 5000) <= 0) return;
            int fd = accept(_socket, nullptr, nullptr);
            if (fd < 0) return;
            std::string request;
            char chunk[1024];
            while (request.find("\r\n\r\n") == std::string::npos)
            {
                auto n = recv(fd, chunk, sizeof(chunk), 0);
                if (n <= 0) break;
                request.append(chunk, n);
            }
            EXPECT_NE(request.find("GET /model%2F1?b=2&a=1 HTTP/"), std::string::npos);
            EXPECT_NE(request.find("Range: bytes=7-9\r\n"), std::string::npos);
            send(fd, reply.data(), reply.size(), MSG_NOSIGNAL);
            sent.set_value();
            if (!tail.empty())
            {
                _release.get_future().wait_for(5s);
                send(fd, tail.data(), tail.size(), MSG_NOSIGNAL);
            }
            close(fd);
        });
    }

    void finish() { std::call_once(_finished, [this] { _release.set_value(); }); }

    ~Server()
    {
        finish();
        _thread.join();
        close(_socket);
    }

    std::string url;
    std::promise<void> sent;

 private:
    int _socket;
    std::thread _thread;
    std::promise<void> _release;
    std::once_flag _finished;
};

class HttpClient : public ::testing::Test
{
 protected:
    static void SetUpTestSuite() { ASSERT_EQ(curl_global_init(CURL_GLOBAL_DEFAULT), CURLE_OK); }
    static void TearDownTestSuite() { curl_global_cleanup(); }
    Client client{{nullptr, 1024, nullptr, 0, 4}};

    api::ObjectCompletionEvent_t next()
    {
        api::ObjectCompletionEvent_t event{};
        unsigned count = 0;
        EXPECT_EQ(client.wait(&event, 1, count, api::OBJECT_WAIT_MODE_BLOCK), ResponseCode::Success);
        EXPECT_EQ(count, 1U);
        return event;
    }
};

const std::string partial = "HTTP/1.1 206 Partial Content\r\n";
const std::string valid = "Content-Range: bytes 7-9/10\r\n";

TEST_F(HttpClient, ValidatesFinalHeadersAndBoundsEveryWrite)
{
    struct Case { std::string reply; bool success; };
    for (const auto & test : std::vector<Case>{
        {"HTTP/1.1 103 Early Hints\r\n\r\n" + partial + valid + "\r\nabc", true},
        {partial + "content-range: Bytes 7-9/*\r\n\r\nabc", true},
        {partial + "\r\nabc", false},
        {partial + "Content-Range: bytes 6-8/10\r\n\r\nabc", false},
        {partial + "Content-Range: bytes 7-9/9\r\n\r\nabc", false},
        {partial + valid + valid + "\r\nabc", false},
        {partial + valid + "\r\nab", false},
        {partial + valid + "\r\nabcd", false},
        {"HTTP/1.1 200 OK\r\n\r\nabcdef", false},
        {"HTTP/1.1 401 Unauthorized\r\n\r\n", false},
        {"HTTP/1.1 403 Forbidden\r\n\r\n", false},
        {"HTTP/1.1 416 Range Not Satisfiable\r\n\r\n", false}})
    {
        SCOPED_TRACE(test.reply);
        Server server(test.reply);
        std::string buffer = "#...#";
        ASSERT_EQ(client.read(server.url.c_str(), {7, 3}, &buffer[1], 42), ResponseCode::Success);
        const auto event = next();
        EXPECT_EQ(event.request_id, 42U);
        EXPECT_EQ(event.response_code == ResponseCode::Success, test.success);
        EXPECT_EQ(buffer.front(), '#');
        EXPECT_EQ(buffer.back(), '#');
        EXPECT_EQ(event.bytes_transferred, 3U - std::count(buffer.begin(), buffer.end(), '.'));
        if (test.success) EXPECT_EQ(buffer, "#abc#");
    }
}

TEST_F(HttpClient, ZeroLengthAndOverflow)
{
    EXPECT_EQ(client.read("unused", {SIZE_MAX, 0}, nullptr, 5), ResponseCode::Success);
    auto event = next();
    EXPECT_EQ(event.response_code, ResponseCode::Success);
    EXPECT_EQ(event.bytes_transferred, 0U);
    char buffer;
    EXPECT_EQ(client.read("unused", {SIZE_MAX, 2}, &buffer, 6), ResponseCode::InvalidParameterError);
}

TEST_F(HttpClient, ConcurrentCompletionsKeepTheirRequestIDs)
{
    Server ok(partial + valid + "Content-Length: 3\r\n\r\na", "bc");
    Server bad("HTTP/1.1 403 Forbidden\r\n\r\n");
    std::string a = "...", b = "...";
    ASSERT_EQ(client.read(ok.url.c_str(), {7, 3}, a.data(), 55), ResponseCode::Success);
    ASSERT_EQ(ok.sent.get_future().wait_for(3s), std::future_status::ready);
    ASSERT_EQ(client.read(bad.url.c_str(), {7, 3}, b.data(), 66), ResponseCode::Success);
    auto second = next();  // The first request is still waiting for its body.
    ok.finish();
    auto first = next();
    EXPECT_EQ(first.request_id, 55U);
    EXPECT_EQ(first.response_code, ResponseCode::Success);
    EXPECT_EQ(first.bytes_transferred, 3U);
    EXPECT_EQ(second.request_id, 66U);
    EXPECT_NE(second.response_code, ResponseCode::Success);
    EXPECT_EQ(second.bytes_transferred, 0U);
    EXPECT_EQ(a, "abc");
    EXPECT_EQ(b, "...");
}

TEST_F(HttpClient, CancelWakesWaiterAndFencesDestination)
{
    std::string buffer = "...";
    {
        Server slow(partial + valid + "Content-Length: 3\r\n\r\na", "bc");
        ASSERT_EQ(client.read(slow.url.c_str(), {7, 3}, buffer.data(), 77), ResponseCode::Success);
        ASSERT_EQ(slow.sent.get_future().wait_for(3s), std::future_status::ready);
        auto waiter = std::async(std::launch::async, [this] { return next(); });
        client.stop();
        auto event = waiter.get();
        EXPECT_EQ(event.request_id, 77U);
        EXPECT_NE(event.response_code, ResponseCode::Success);
        EXPECT_EQ(event.bytes_transferred, buffer == "a.." ? 1U : 0U);
        buffer = "new";
    }
    EXPECT_EQ(buffer, "new");
}

TEST_F(HttpClient, ReleaseWithReadInFlight)
{
    std::string buffer = "...";
    {
        Server slow(partial + valid + "Content-Length: 3\r\n\r\n", "abc");
        ASSERT_EQ(client.read(slow.url.c_str(), {7, 3}, buffer.data(), 88), ResponseCode::Success);
        ASSERT_EQ(slow.sent.get_future().wait_for(3s), std::future_status::ready);
        client.release();
        buffer = "new";
    }
    EXPECT_EQ(buffer, "new");
}

} // namespace
} // namespace runai::llm::streamer::impl::http
