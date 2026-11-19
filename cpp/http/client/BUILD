# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client",
    deps = [
        "@curl",
        "//common/client_mgr",
        "//utils/threadpool",
    ],
)

runai_cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [":client"],
)
