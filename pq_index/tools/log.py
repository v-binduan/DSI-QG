#!/usr/bin/env python
# -*- coding: gbk -*-

import logging
import os
import sys

#os.system('mkdir log')
logger = logging.getLogger('gdm')
logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('./log/gdm.log.debug', 'a')
fmt = logging.Formatter(
        '%(levelname)s: %(asctime)s %(process)d %(threadName)s[%(thread)d]'
        ' [%(filename)s:%(lineno)s][%(funcName)s] %(message)s')
debug_handler.setFormatter(fmt)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)

info_handler = logging.FileHandler('./log/gdm.log', 'a')
fmt = logging.Formatter('%(asctime)s %(threadName)s[%(thread)d] %(message)s')
info_handler.setFormatter(fmt)
info_handler.setLevel(logging.INFO)
logger.addHandler(info_handler)

fmt = logging.Formatter('%(levelname)s %(asctime)s [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s')
stdout = logging.StreamHandler(stream=sys.stdout)
stdout.setFormatter(fmt)
stdout.setLevel(logging.INFO)
logger.addHandler(stdout)

fmt = logging.Formatter('%(levelname)s %(asctime)s [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s')
stderr = logging.StreamHandler(stream=sys.stderr)
stderr.setFormatter(fmt)
stderr.setLevel(logging.WARNING)
logger.addHandler(stderr)
