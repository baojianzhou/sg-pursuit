# -*- coding: utf-8 -*-
__all__ = ["get_list_files"]
import os


def get_list_files(input_folder, prefix=None):
    if not os.path.exists(input_folder):
        file_list = []
        return file_list
    for (dir_path, dir_names, file_names) in os.walk(input_folder):
        file_list = sorted([os.path.join(input_folder, filename)
                            for filename in file_names])
        if prefix is None or prefix == '':
            return file_list
        file_list = [_ for _ in file_list if _.find(prefix) != -1]
        return file_list
