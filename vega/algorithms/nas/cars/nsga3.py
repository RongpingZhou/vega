# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Function for pNSGA-III."""
import numpy as np
import random
from zeus.report import NonDominatedSorting


def CARS_NSGA(target, objs, N):
    """pNSGA-III (CARS-NSGA).

    :param target: the first objective, e.g. accuracy
    :type target: array
    :param objs: the other objective, e.g. FLOPs, number of parameteres
    :type objs: array
    :param N: number of population
    :type N: int
    :return: The selected samples
    :rtype: array
    """
    selected = np.zeros(target.shape[0])
    Fs = []
    for obj in objs:
        Fs.append(NonDominatedSorting(np.vstack((1 / target, obj))))
        Fs.append(NonDominatedSorting(np.vstack((1 / target, 1 / obj))))
    stage = 0
    while (np.sum(selected) < N):
        current_front = []
        for i in range(len(Fs)):
            if len(Fs[i]) > stage:
                current_front.append(Fs[i][stage])
        current_front = [np.array(c) for c in current_front]
        current_front = np.hstack(current_front)
        current_front = list(set(current_front))
        if np.sum(selected) + len(current_front) <= N:
            for i in current_front:
                selected[i] = 1
        else:
            not_selected_indices = np.arange(len(selected))[selected == 0]
            crt_front = [index for index in current_front if index in not_selected_indices]
            num_to_select = N - np.sum(selected).astype(np.int32)
            current_front = crt_front if len(crt_front) <= num_to_select else random.sample(crt_front, num_to_select)
            for i in current_front:
                selected[i] = 1
        stage += 1
    return np.where(selected == 1)[0]
