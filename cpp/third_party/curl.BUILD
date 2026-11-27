# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dynamic_link",
    define_values = {"USE_SYSTEM_LIBS": "1"},
)

[cc_library(
    name = "curl_" + arch,
    hdrs = glob([arch + "-curl/include/curl/*.h"], allow_empty = True),
    includes = [arch + "-curl/include"],
    linkopts = select({
        ":dynamic_link": ["-lcurl"],
        "//conditions:default": [
            "-L/opt/" + arch + "-curl/lib",
            "-L/opt/" + arch + "-ssl/lib",
            "-L/opt/" + arch + "-zlib/lib",
            "-l:libcurl.a", "-l:libssl.a", "-l:libcrypto.a", "-l:libz.a",
            "-ldl", "-lpthread",
        ],
    }),
) for arch in ["aarch64", "x86_64"]]

alias(
    name = "curl",
    actual = select({
        "@platforms//cpu:aarch64": ":curl_aarch64",
        "@platforms//cpu:x86_64": ":curl_x86_64",
    }),
)
