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
from . import utils
from .log import setup_main_logger, log_sockeye_version

logger = logging.getLogger(__name__)


def main():
    """
    Commandline interface to rename or copy parameters.
    """
    setup_main_logger(console=True, file_logging=False)
    params = argparse.ArgumentParser(description="Rename or copy specific parameters.")
    arguments.add_rename_parameters_args(params)
    args = params.parse_args()
    utils.check_condition(
        len(args.rename_from) == len(args.rename_to),
        "Please provide a new name for all parameters in --rename-from"
    )
    rename_parameters(args)


def rename_parameters(args: argparse.Namespace):
    log_sockeye_version(logger)

    if os.path.isdir(args.params_in):
        param_path = os.path.join(args.params_in, C.PARAMS_BEST_NAME)
    else:
        param_path = args.params_in

    params = mx.nd.load(param_path)
    for from_name, to_name in zip(args.rename_from, args.rename_to):
        if from_name in params:
            logger.info("\tFound '%s': shape=%s", from_name, str(params[from_name].shape))
            if to_name in params:
                logger.info("\tOverwriting '%s': shape=%s", to_name, str(params[to_name].shape))
            else:
                logger.info("\tCreating '%s': shape=%s", to_name, str(params[to_name].shape))
            params[to_name] = params[from_name]
            if not args.copy:
                logger.info("\tRemoving '%s': shape=%s", from_name, str(params[from_name].shape))
                del params[from_name]
        else:
            logger.info("\tDid not find %s", from_name)

    logger.info("Writing extracted parameters to '%s'", args.params_out)
    mx.nd.save(args.params_out, params)


if __name__ == "__main__":
    main()
