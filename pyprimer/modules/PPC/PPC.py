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

        summary_col_list = ["Primer Group",
                            "F Version",
                            "R Version",
                            "Mean PPC",
                            "Sequences matched(%)"]
        summary = pd.DataFrame(columns=summary_col_list)

        if self.memsave:
            self._saver.initialize(bench_df)

        # ugly
        def helper(sequences, Fs, Rs, COL_LIST, deletions=0, insertions=0, substitutions=2):
            df = pd.DataFrame(columns=self.COL_LIST)
            for f in Fs:
                for r in Rs:
                    header = sequences[0]
                    f_name = f[2]
                    r_name = r[2]
                    f_ver = f[5]
                    r_ver = r[5]
                    f_res = TOOLS.match_fuzzily(
                        f_ver, sequences[1], deletions, insertions, substitutions)
                    r_res = TOOLS.match_fuzzily(
                        r_ver, sequences[2], deletions, insertions, substitutions)

                    if type(f_res) == type(tuple()):
                        start = f_res[0]
                        f_match = f_ver

                    elif f_res == None:
                        start = None
                        f_match = ""

                    else:
                        start = f_res.start
                        f_match = f_res.matched

                    if type(r_res) == type(tuple()):
                        r_start = r_res[0]
                        end = (len(sequences[1]) - 1) - r_start
                        r_match = r_ver

                    elif r_res == None:
                        end = None
                        r_match = ""
                    else:
                        end = (len(sequences[1]) - 1) - r_res.start
                        r_match = r_res.matched

                    if start == None or end == None:
                        amplicon = ""
                        amplicon_length = 0

                    else:
                        amplicon = sequences[1][start:end]
                        amplicon_length = len(amplicon)

                    PPC = TOOLS.calculate_PPC(F_primer=f_ver,
                                              F_match=f_match,
                                              R_primer=r_ver,
                                              R_match=r_match)

                    df.loc[len(df)] = [f_name, f_ver, r_name, r_ver,
                                       header, amplicon, amplicon_length, start, end, PPC]
            return df

        with tqdm(total=100, file=sys.stdout) as pbar:
            for group in unique_groups:
                # print("Processing group {} against {} sequences".format(
                #     group, self.sequences.shape[0]))
                Fs = self.primers.loc[(self.primers["ID"] == group) & (
                    self.primers["Type"] == "F"), :].values
                Rs = self.primers.loc[(self.primers["ID"] == group) & (
                    self.primers["Type"] == "R"), :].values

                dsequences = dd.from_pandas(self.sequences, npartitions=nCores)
                df_series = dsequences.map_partitions(
                    lambda df: df.apply(
                        lambda x: helper(x, Fs, Rs, self.COL_LIST, deletions, insertions, substitutions), axis=1), meta=('df', None)).compute(scheduler='processes')

                group_df = pd.concat(df_series.tolist())

                # group_df.to_csv("{}.csv".format(group), index = False)

                v_stats = dict((key, []) for key in summary_col_list)
                for fversion in group_df["F Primer Version"].unique():
                    for rversion in group_df["R Primer Version"].unique():
                        mean_ppc = group_df.loc[(group_df["F Primer Version"] == fversion) & (
                            group_df["R Primer Version"] == rversion), "PPC"].mean()
                        seqs_matched = len(group_df.loc[(group_df["F Primer Version"] == fversion) & (
                            group_df["R Primer Version"] == rversion) & (group_df["Amplicon Sense Length"] != 0), "Amplicon Sense Length"])
                        n_seqs = len(group_df.loc[(group_df["F Primer Version"] == fversion) & (
                            group_df["R Primer Version"] == rversion), "Amplicon Sense Length"])
                        v_stats["Primer Group"].append(group)
                        v_stats["F Version"].append(fversion)
                        v_stats["R Version"].append(rversion)
                        v_stats["Mean PPC"].append(mean_ppc)
                        v_stats["Sequences matched(%)"].append(
                            (seqs_matched / n_seqs)*100)
                group_stats = pd.DataFrame(v_stats, columns=summary_col_list)
                # group_stats.to_csv("{}_stats.csv".format(group), index = False)
                summary = summary.append(group_stats)

                if self.memsave:
                    self._saver.save_group(group_df, group)
                else:
                    bench_df = bench_df.append(group_df)
                pbar.update(1/len(unique_groups)*100)

        if self.memsave:
            print("Extended benchmark results were written to {}".format(
                os.path.join(self._saver.tempdir, "PCRBenchmark.h5")))
        
        return summary