import subprocess
import os
import tempfile
import time
from typing import AnyStr, Dict, Tuple

CURRENT_DIR = os.path.dirname(__file__)

class Coma:

    def __init__(self,
                 max_n: int = 0,
                 strategy: str = "COMA_OPT",
                 java_xmx: str = "1024m"):
        self.__max_n = int(max_n)
        self.__strategy = strategy
        self.__java_XmX = java_xmx

    def run_coma_jar(self, source_table_f_name: str, target_table_f_name: str, coma_output_path: str,
                     tmp_folder_path: str) -> str:
        """
        runs coma and returns the filename in output
        """
        jar_path = os.path.join(CURRENT_DIR, 'coma.jar')
        source_data = os.path.join(tmp_folder_path, source_table_f_name)
        target_data = os.path.join(tmp_folder_path, target_table_f_name)
        coma_output_path = os.path.join(tmp_folder_path, coma_output_path)
        with open(os.path.join(tmp_folder_path, "NUL"), "w") as fh:
            subprocess.call(['java', f'-Xmx{self.__java_XmX}',
                             '-cp', jar_path,
                             '-DinputFile1=' + source_data,
                             '-DinputFile2=' + target_data,
                             '-DoutputFile=' + coma_output_path,
                             '-DmaxN=' + str(self.__max_n),
                             '-Dstrategy=' + self.__strategy,
                             'Main'], stdout=fh, stderr=fh)
        os.remove(source_data)
        os.remove(target_data)
        return coma_output_path

    @staticmethod
    def __get_column(match) -> str:
        return ".".join(match.split(".")[1:])
