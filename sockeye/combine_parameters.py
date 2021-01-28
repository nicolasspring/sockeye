# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Rename specific parameters.
"""

import argparse
import os
import logging

import mxnet as mx

from . import arguments
from . import constants as C
from .log import setup_main_logger, log_sockeye_version

logger = logging.getLogger(__name__)


def main():
    """
    Commandline interface to combine parameter files.
    """
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description="Combine parameter from multiple files.")
    arguments.add_combine_parameters_args(params)
    args = params.parse_args()
    combine_parameters(args)


def combine_parameters(args: argparse.Namespace):
    log_sockeye_version(logger)

    out_params = {}
    for inpath in args.params_in:
        if os.path.isdir(inpath):
            param_path = os.path.join(inpath, C.PARAMS_BEST_NAME)
        else:
            param_path = inpath

        logger.info("\tLoading params from %s", param_path)
        new_params = mx.nd.load(param_path)
        out_params = {**out_params, **new_params}

    logger.info("\tSaving combined params to %s", args.params_out)
    mx.nd.save(args.params_out, out_params)


if __name__ == "__main__":
    main()
