# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 boomb0om
# SPDX-License-Identifier: Apache-2.0

load("//:rules.bzl", "runai_cc_auto_library", "runai_cc_test")

runai_cc_auto_library(
    name = "client",
    deps = [
        "@curl",
        "//common/backend_api/object_storage",
        "//common/client_mgr",
        "//utils/env",
        "//utils/scope_guard",
        "//utils/threadpool",
    ],
)

runai_cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [":client"],
)
