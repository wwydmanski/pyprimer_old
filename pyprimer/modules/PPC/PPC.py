# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from numba import jit, prange
from tqdm import trange
from tqdm import tqdm
import h5py

import operator
import dask.dataframe as dd
import dask.multiprocessing
import dask.threaded
import sys
import time
import warnings
from .ppc_tools import TOOLS, _MemSaver

class PPC(object):
    COL_LIST = ["F Primer Name",
                "F Primer Version",
                "R Primer Name",
                "R Primer Version",
                "Sequence Header",
                "Amplicon Sense",
                "Amplicon Sense Length",
                "Amplicon Sense Start",
                "Amplicon Sense End",
                "PPC"]

    SUMMARY_COL_LIST = ["Primer Group",
                        "F Version",
                        "R Version",
                        "Mean PPC",
                        "Sequences matched(%)"]

    def __init__(self, primer_df, 
                        sequence_df,
                        memsave=False,
                        tempdir="./tmp/",
                        fname="PCRBenchmark.h5"):
        self.primers = primer_df
        self.sequences = sequence_df
        self.memsave = memsave
        if memsave:
            self._saver = _MemSaver(tempdir, fname, self.COL_LIST)

    def analyse_primers(self,
                        deletions=0,
                        insertions=0,
                        substitutions=2,
                        nCores=2) -> pd.DataFrame:

        unique_groups = self.primers["ID"].unique()

        bench_df = pd.DataFrame(columns=self.COL_LIST)
        summary = pd.DataFrame(columns=self.SUMMARY_COL_LIST)

        if self.memsave:
            self._saver.initialize(bench_df)

        with tqdm(unique_groups) as pbar:
            for group in pbar:
                filter_group = self.primers["ID"] == group
                filter_forward = self.primers["Type"] == "F"
                filter_reverse = self.primers["Type"] == "R"

                Fs = self.primers.loc[filter_group & filter_forward].values
                Rs = self.primers.loc[filter_group & filter_reverse].values

                dsequences = dd.from_pandas(self.sequences, npartitions=nCores)
                df_series = dsequences.map_partitions(
                    lambda df: df.apply(
                        lambda x: self.helper(x, Fs, Rs, deletions, insertions, substitutions), axis=1), meta=('df', None)).compute(scheduler='processes')

                group_df = pd.concat(df_series.tolist())

                v_stats = dict((key, []) for key in self.SUMMARY_COL_LIST)
                for fversion in group_df["F Primer Version"].unique():
                    for rversion in group_df["R Primer Version"].unique():
                        filter_r_version = (group_df["R Primer Version"] == rversion)
                        filter_f_version = (group_df["F Primer Version"] == fversion)
                        filter_matching = filter_r_version & filter_f_version
                        
                        n_seqs = np.sum(filter_matching)
                        seqs_matched = np.sum(filter_matching & (group_df["Amplicon Sense Length"] != 0))

                        mean_ppc = group_df.loc[filter_matching, "PPC"].mean().round(5)
                        
                        v_stats["Primer Group"].append(group)
                        v_stats["F Version"].append(fversion)
                        v_stats["R Version"].append(rversion)
                        v_stats["Mean PPC"].append(mean_ppc)
                        v_stats["Sequences matched(%)"].append(
                            (seqs_matched / n_seqs)*100)
                group_stats = pd.DataFrame(v_stats, columns=self.SUMMARY_COL_LIST)
                summary = summary.append(group_stats)

                if self.memsave:
                    self._saver.save_group(group_df, group)
                else:
                    bench_df = bench_df.append(group_df)

        if self.memsave:
            print("Extended benchmark results were written to {}".format(
                os.path.join(self._saver.tempdir, "PCRBenchmark.h5")))
        
        return summary

    def helper(self, sequences, Fs, Rs, deletions, insertions, substitutions):
        res = []
        for f in Fs:
            for r in Rs:
                header = sequences[0]
                f_name = f[2]
                r_name = r[2]
                f_ver = f[5]
                r_ver = r[5]

                start, f_match = TOOLS.match_fuzzily(
                    f_ver, sequences[1], deletions, insertions, substitutions)
                r_start, r_match = TOOLS.match_fuzzily(
                    r_ver, sequences[2], deletions, insertions, substitutions)

                try:
                    end = len(sequences[1]) - 1 - r_start
                except TypeError:
                    end = None
                    
                if start is None or end is None:
                    amplicon = ""
                    amplicon_length = 0
                else:
                    amplicon = sequences[1][start:end]
                    amplicon_length = len(amplicon)

                PPC = TOOLS.calculate_PPC(F_primer=f_ver,
                                            F_match=f_match,
                                            R_primer=r_ver,
                                            R_match=r_match)
                res.append([f_name, f_ver, r_name, r_ver,
                                    header, amplicon, amplicon_length, start, end, PPC])

        df = pd.DataFrame(res, columns=self.COL_LIST)
        return df