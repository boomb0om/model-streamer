# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


# Install next to libstreamer.so so its $ORIGIN search path also works in a virtualenv.
package = "runai_model_streamer.libstreamer.lib"
library = Path(__file__).parent.joinpath(*package.split("."), "libstreamerhttp.so")
if not library.is_file():
    raise FileNotFoundError(f"Build {library.resolve()} before packaging the HTTP plugin")

setup(
    name="runai-model-streamer-http",
    version=os.getenv("PACKAGE_VERSION", "0.0.0"),
    # attribution: ships NOTICE (third-party notices, kept current by the Renovate pipeline)
    license_files=("LICENSE", "NOTICE"),
    packages=[package],
    package_data={package: [library.name]},
    cmdclass={"bdist_wheel": bdist_wheel},
)
