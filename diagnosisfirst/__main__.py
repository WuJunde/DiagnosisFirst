# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m diagnosisfirst方式直接执行。

Authors: wujunde(wujunde@baidu.com)
Date:    2021/11/22 21:17:20
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from diagnosisfirst.cmdline import main
sys.exit(main())
