/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/storage_uri/storage_uri.h"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>

#include "utils/random/random.h"

namespace runai::llm::streamer::common::s3
{

TEST(Uri, Empty)
{
    std::unique_ptr<StorageUri> uri;
    EXPECT_THROW(uri = std::make_unique<StorageUri>(""), std::exception);
}

TEST(Uri, HttpPreservesOriginalUrlAndRedactsItsQuery)
{
    for (const std::string scheme : {"http", "https"})
    {
        const auto url = scheme + "://host:8080/a%2Fb?sig=secret&x=1";
        StorageUri uri(url);
        EXPECT_TRUE(uri.is_http());
        EXPECT_EQ(uri.uri, url);
        EXPECT_EQ(uri.bucket, "host:8080");
        EXPECT_EQ(uri.path, "a%2Fb");
        std::ostringstream log;
        log << uri;
        EXPECT_EQ(log.str().find("secret"), std::string::npos);
        EXPECT_EQ(display_path(url), scheme + "://host:8080/a%2Fb");
    }
    EXPECT_EQ(display_path("/local/file?name"), "/local/file?name");
}

TEST(Uri, Valid_S3_Path)
{
    auto bucket = utils::random::string();
    auto path = utils::random::string();
    auto s3_path = "s3://" + bucket + "/" + path;
    std::unique_ptr<StorageUri> uri;
    EXPECT_NO_THROW(uri = std::make_unique<StorageUri>(s3_path));
    EXPECT_EQ(uri->scheme, "s3");
    EXPECT_EQ(uri->bucket, bucket);
    EXPECT_EQ(uri->path, path);
}

TEST(Uri, Valid_GCS_Path)
{
    auto bucket = utils::random::string();
    auto path = utils::random::string();
    auto s3_path = "gs://" + bucket + "/" + path;
    std::unique_ptr<StorageUri> uri;
    EXPECT_NO_THROW(uri = std::make_unique<StorageUri>(s3_path));
    EXPECT_EQ(uri->scheme, "gs");
    EXPECT_EQ(uri->bucket, bucket);
    EXPECT_EQ(uri->path, path);
}

TEST(Valid, Empty_Path)
{
    auto bucket = utils::random::string();
    auto s3_path = "s3://" + bucket + "/";
    std::unique_ptr<StorageUri> uri;
    EXPECT_THROW(uri = std::make_unique<StorageUri>(s3_path), std::exception);
}

TEST(Valid, Empty_S3_Path)
{
    std::unique_ptr<StorageUri> uri;
    EXPECT_THROW(uri = std::make_unique<StorageUri>("s3://"), std::exception);
}

TEST(Valid, Empty_Bucket)
{
    auto path = utils::random::string();
    auto s3_path = "s3:///" + path;
    std::unique_ptr<StorageUri> uri;
    EXPECT_THROW(uri = std::make_unique<StorageUri>(s3_path), std::exception);
}

TEST(Invalid, InvalidScheme)
{
    auto path = utils::random::string();
    auto endpoint = "nfs://" + path;
    std::unique_ptr<StorageUri> uri;
    EXPECT_THROW(uri = std::make_unique<StorageUri>(endpoint), std::exception);
}

}; // namespace runai::llm::streamer::common::s3
