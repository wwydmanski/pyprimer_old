from pyprimer.utils.sequence import PCRPrimer, Sequence, READ_MODES
from pyprimer.utils.sequence import Sequence
from pyprimer.modules.PPC import PPC
import sys

data_dir = "/bi/aim/scratch/afrolova/COVID19/github/pyprimer/data"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("Reading primers...")
    test_primer = PCRPrimer(READ_MODES.DIRECTORY)
    primer_df = test_primer.describe_primers(f"{data_dir}/primers")
    print("Primers read")
    print("Reading sequences...")
    test_sequence = Sequence(READ_MODES.FASTA)
    sequence_df = test_sequence.describe_sequences(f"{data_dir}/merged_latest.fasta", verbose=True)
    print("Sequences read")

    test_pcr = PPC(primer_df, sequence_df)
    summary = test_pcr.analyse_primers(nCores=8, deletions=1, insertions=0, substitutions=2)
    summary.to_csv("summary_df.csv", index = False)