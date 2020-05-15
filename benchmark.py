# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from numba import jit
import h5py
from tqdm import trange

class TOOLS:


class PCR(object):
    
    def __init__(self, primer_df, sequence_df):
        self.primers = primer_df
        self.sequences = sequence_df.values
    
    def analyse_primers(self, memsave = False, tempdir = "./tmp/", fname = "PCRBenchmark.h5"):
        self.memsave = memsave
        self.tempdir = tempdir
        self.fname = fname
        unique_groups = self.primers["ID"].unique()
        col_list = ["F Primer Name",
                    "F Primer Version",
                    "R Primer Name",
                    "R Primer Version",
                    "Sequence Header",
                    "Best Amplicon",
                    "Best Amplicon Start",
                    "Best Amplicon End",
                    "Bind posiitons Sense",
                    "Bind positions Antisense",
                    "PPC"]
        bench_df = pd.DataFrame(columns = col_list)
        self.bench = bench_df
        if self.memsave:
            os.makedirs(self.tempdir, exist_ok = True)
            bench_df.to_hdf(path_or_buf=os.path.join(self.tempdir,"PCRBenchmark.h5"),
                            mode = "w",
                            format = "table",
                            data_columns = True)
            del bench_df
        for group in trange(len(unique_groups)):
            Fs = self.primers.loc[self.primers["ID"] == group &
                                  self.primers["Type" == "F",],:].values
            Rs = self.primers.loc[self.primers["ID"] == group &
                                  self.primers["Type" == "R",],:].values
    
    def analyse_probes(self, memsave = False, tempdir = "./tmp/")


            
            





