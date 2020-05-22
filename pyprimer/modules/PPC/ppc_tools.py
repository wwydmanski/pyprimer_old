import os
from fuzzysearch import find_near_matches
from fuzzywuzzy import fuzz
import numpy as np
import warnings
import tables


class TOOLS:
    def match_fuzzily(pattern,
                      sequence,
                      deletions=0,
                      insertions=0,
                      substitutions=2):

        if pattern in sequence:
            start = sequence.index(pattern)
            # end = start + len(pattern)
            return start, pattern
        else:
            result = find_near_matches(pattern,
                                       sequence,
                                       max_substitutions=substitutions,
                                       max_insertions=insertions,
                                       max_deletions=deletions)
            if len(result) != 0:
                idx_min = np.argmin([i.dist for i in result])
                return result[idx_min].start, result[idx_min].matched
            else:
                return None, ""

    def calculate_PPC(F_primer, F_match, R_primer, R_match):
        Fl = float(len(F_primer))
        Fm = np.round((fuzz.ratio(F_primer, F_match) / 100) * Fl)
        Rl = float(len(R_primer))
        Rm = np.round((fuzz.ratio(R_primer, R_match) / 100) * Rl)
        sigma_m = np.std([Fm, Rm])
        mi_m = np.mean([Fm, Rm])
        if mi_m == 0:
            PPC = 0
            return PPC
        CV_m = sigma_m / mi_m
        PPC = (Fm/Fl) * (Rm/Rl) * (1-CV_m)
        if PPC == np.nan:
            PPC = 0
        return PPC


class _MemSaver:
    def __init__(self, tempdir, fname, col_list):
        self.tempdir = tempdir
        self.fname = fname
        self.col_list = col_list

    def initialize(self, bench_df):
        os.makedirs(self.tempdir, exist_ok=True)
        bench_df.to_hdf(path_or_buf=os.path.join(self.tempdir, self.fname),
                        key="bench",
                        mode="w",
                        format="table",
                        data_columns=True)
    
    def save_group(self, group_df, key):
        # extra weird fix for this problem:
        # https://stackoverflow.com/questions/60677863/python-pandas-append-dataframe-with-array-content-to-hdf-file
        # https://mlog.club/article/5501050
        for i in range(len(self.col_list)-1):
            group_df[self.col_list[i]
                        ] = group_df[self.col_list[i]].astype(str)

        group_df[self.col_list[len(
            self.col_list)-1]] = group_df[self.col_list[len(self.col_list)-1]].astype(float)
        warnings.filterwarnings(
            'ignore', category=tables.NaturalNameWarning)

        group_df.to_hdf(path_or_buf=os.path.join(self.tempdir, self.fname),
                        key=key,
                        mode="a",
                        format="table",
                        data_columns=True)