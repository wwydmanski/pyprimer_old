import numpy as np
import pytest
from pyprimer.modules import PPC

data_dir = "/bi/aim/scratch/afrolova/COVID19/github/pyprimer/data"

@pytest.fixture()
def read_squences():
    from pyprimer.utils.sequence import PCRPrimer, Sequence, READ_MODES
    test_sequence = Sequence(READ_MODES.FASTA)
    sequences_df = test_sequence.describe_sequences(f"{data_dir}/merged.fasta")

    return sequences_df

def test_design_points(read_squences):
    test_pcr = PPC(None, read_squences)

    # CDC_2019-nCoV_N2
    F = "TTACAAACATTGGCCGCAAA"
    R = "GCGCGACATTCCGAAGAA"
    P = "CTAGCCATGCCCTTAGT"

    # Control_BVDV
    F2 = "TCAGCGAAGGCCGAAAAG"
    R2 = "TGCTACCCCCTCCATTATGC"
    P2 = "CTAGCCATGCCCTTAGT"

    ppc, matches = test_pcr.get_primer_metrics([F, F2], [R, R2], [P, P2], nCores=2)
    assert np.isclose(ppc[0], 0.94136)
    assert np.isclose(matches[0], 99.39759036)

    assert np.isclose(ppc[1], 0.)
    assert np.isclose(matches[1], 0.)

if __name__ == "__main__":
    test_design_points(read_squences())