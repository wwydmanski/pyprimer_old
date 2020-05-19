# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from numba import jit, prange
from tqdm import trange
from tqdm import tqdm
import h5py
from fuzzysearch import find_near_matches
from fuzzywuzzy import fuzz
import operator
import dask.dataframe as dd
import dask.multiprocessing
import dask.threaded
import sys
import time
import tables
import warnings

class TOOLS:

    def match_fuzzily(pattern_,
                      sequence_,
                      deletions=1,
                      insertions=0,
                      substitutions=2):

        if pattern_ in sequence_:
            start = sequence_.index(pattern_)
            end = start + len(pattern_)
            return (start, end)

        else:

            result = find_near_matches(pattern_,
                                       sequence_,
                                       max_substitutions=substitutions,
                                       max_insertions=insertions,
                                       max_deletions=deletions)
            if len(result) > 1:
                dist_list = np.ndarray(shape=(len(result),))
                for i in range(len(result)):
                    dist_list[i] = result[i].dist
                idx_max = np.where(dist_list == np.max(dist_list))
                return result[idx_max[1][0]]

            elif len(result) == 1:
                return result[0]

            else:
                return None

    def calculate_PPC(F_primer, F_match, R_primer, R_match):
        Fl = float(len(F_primer))
        Fm = np.round((fuzz.ratio(F_primer, F_match) / 100) * Fl)
        Rl = float(len(R_primer))
        Rm = np.round((fuzz.ratio(R_primer, R_match) / 100) * Rl)
        sigma_m = np.std([Fm,Rm])
        mi_m = np.mean([Fm,Rm])
        if mi_m == 0:
            PPC = 0
            return PPC
        CV_m = sigma_m / mi_m
        PPC = (Fm/Fl) * (Rm/Rl) * (1-CV_m)
        if PPC == np.nan:
            PPC = 0
        return PPC


class PCR(object):

    def __init__(self, primer_df, sequence_df):
        self.primers = primer_df
        self.sequences = sequence_df

    # @jit(nopython=False, parallel = True)
    def analyse_primers(self,
                        memsave = False,
                        tempdir = "./tmp/",
                        fname = "PCRBenchmark.h5",
                        deletions = 2,
                        insertions = 0,
                        substitutions = 2,
                        nCores = 2):

        self.memsave = memsave
        self.tempdir = tempdir
        self.fname = fname
        self.deletions = deletions
        self.instertions = insertions
        self.substitutions = substitutions
        self.nCores = nCores

        unique_groups = self.primers["ID"].unique()
        col_list = ["F Primer Name",
                    "F Primer Version",
                    "R Primer Name",
                    "R Primer Version",
                    "Sequence Header",
                    "Amplicon Sense",
                    "Amplicon Sense Length",
                    "Amplicon Sense Start",
                    "Amplicon Sense End",
                    "PPC"]
        bench_df = pd.DataFrame(columns=col_list)

        summary_col_list = ["Primer Group",
                            "F Version",
                            "R Version",
                            "Mean PPC",
                            "Sequences matched(%)"]
        summary = pd.DataFrame(columns=summary_col_list)

        self.bench = bench_df

        if self.memsave:
            os.makedirs(self.tempdir, exist_ok=True)
            bench_df.to_hdf(path_or_buf=os.path.join(self.tempdir, self.fname),
                            key="bench",
                            mode="w",
                            format="table",
                            data_columns=True)
            # del bench_df

        # ugly
        def helper(sequences, Fs, Rs, col_list):
            df = pd.DataFrame(columns = col_list)
            for f in Fs:
                for r in Rs:
                    header = sequences[0]
                    f_name = f[2]
                    r_name = r[2]
                    f_ver = f[5]
                    r_ver = r[5]
                    f_res = TOOLS.match_fuzzily(f_ver, sequences[1])
                    r_res = TOOLS.match_fuzzily(r_ver, sequences[2])

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

                    PPC = TOOLS.calculate_PPC(F_primer = f_ver,
                                                F_match = f_match,
                                                R_primer = r_ver,
                                                R_match = r_match)

                    df.loc[len(df)] = [f_name, f_ver, r_name, r_ver, header, amplicon, amplicon_length, start, end, PPC]
            return df

        with tqdm(total=100, file=sys.stdout) as pbar:
            for group in unique_groups:
                print("Processing group {} against {} sequences".format(group, self.sequences.shape[0]))
                Fs = self.primers.loc[(self.primers["ID"] == group) & (self.primers["Type"] == "F"),:].values
                Rs = self.primers.loc[(self.primers["ID"] == group) & (self.primers["Type"] == "R"),:].values

                dsequences = dd.from_pandas(self.sequences, npartitions=nCores)
                df_series = dsequences.map_partitions(
                    lambda df: df.apply(
                        lambda x: helper(x, Fs, Rs, col_list), axis=1), meta=('df', None)).compute(scheduler='processes')

                group_df = pd.concat(df_series.tolist())


                # group_df.to_csv("{}.csv".format(group), index = False)

                v_stats = dict((key,[]) for key in summary_col_list)
                for fversion in group_df["F Primer Version"].unique():
                    for rversion in group_df["R Primer Version"].unique():
                        mean_ppc = group_df.loc[(group_df["F Primer Version"] == fversion) & (group_df["R Primer Version"] == rversion), "PPC"].mean()
                        seqs_matched = len(group_df.loc[(group_df["F Primer Version"] == fversion) & (group_df["R Primer Version"] == rversion) & (group_df["Amplicon Sense Length"] != 0), "Amplicon Sense Length"])
                        n_seqs = len(group_df.loc[(group_df["F Primer Version"] == fversion) & (group_df["R Primer Version"] == rversion), "Amplicon Sense Length"])
                        v_stats["Primer Group"].append(group)
                        v_stats["F Version"].append(fversion)
                        v_stats["R Version"].append(rversion)
                        v_stats["Mean PPC"].append(mean_ppc)
                        v_stats["Sequences matched(%)"].append((seqs_matched / n_seqs)*100)
                group_stats = pd.DataFrame(v_stats, columns = summary_col_list)
                # group_stats.to_csv("{}_stats.csv".format(group), index = False)
                summary = summary.append(group_stats)

                if self.memsave:
                    # extra weird fix for this problem:
                    # https://stackoverflow.com/questions/60677863/python-pandas-append-dataframe-with-array-content-to-hdf-file
                    # https://mlog.club/article/5501050
                    for i in range(len(col_list)-1):
                        group_df[col_list[i]] = group_df[col_list[i]].astype(str)

                    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

                    group_df.to_hdf(path_or_buf=os.path.join(self.tempdir, self.fname),
                                    key = group,
                                    mode = "a",
                                    format = "table",
                                    data_columns = True)
                else:
                    self.bench = self.bench.append(group_df)
                pbar.update(1/len(unique_groups)*100)

        if self.memsave:
            print("Extended benchmark results were written to {}".format(
                os.path.join(self.tempdir, "PCRBenchmark.h5")))
        self.summary = summary
    # def analyse_probes(self, memsave = False, tempdir = "./tmp/")

