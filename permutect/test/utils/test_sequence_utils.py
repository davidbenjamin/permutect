from math import isclose

import permutect.utils.sequence_utils as sequence_utils
from permutect.utils.sequence_utils import StemLoop
from permutect.utils.sequence_utils import StemLoopPart
from permutect.utils.sequence_utils import contains_near_stem_loop
from permutect.utils.sequence_utils import find_near_stem_loop
from permutect.utils.sequence_utils import fold_sequence


def make_stem_loop(stacks=None, hairpin=None, contig="chr1"):
    return StemLoop(
        stacks=[] if stacks is None else stacks,
        hairpin=hairpin,
        contig=contig,
        start_position=0,
        sequence="A" * 30,
    )


def test_stem_loop_total_energy_includes_stacks_and_hairpin():
    stem_loop = make_stem_loop(
        stacks=[
            StemLoopPart(bounds=[(7, 11), (18, 22)], energy=-2.8, desc="STACK:CTAAGC/GCTTAG"),
            StemLoopPart(bounds=[(2, 4), (25, 27)], energy=-1.2, desc="STACK:AAA/TTT"),
        ],
        hairpin=StemLoopPart(bounds=[(12, 17)], energy=3.7, desc="HAIRPIN:CT/GG"),
    )

    assert isclose(stem_loop.total_energy, -0.3)


def test_stem_loop_total_energy_without_hairpin():
    stem_loop = make_stem_loop(stacks=[StemLoopPart(bounds=[(7, 11)], energy=-2.8, desc="STACK:AAAA/TTTT")])

    assert stem_loop.total_energy == -2.8


def test_contains_near_checks_each_stack_interval_with_inclusive_distance():
    stem_loop = make_stem_loop(
        stacks=[StemLoopPart(bounds=[(16, 19), (22, 24)], energy=-1.0, desc="STACK:AAA/TTT")],
        hairpin=StemLoopPart(bounds=[(20, 21)], energy=2.0, desc="HAIRPIN:A/T"),
    )

    assert all(stem_loop.contains_near("chr1", position, dist=1) for position in range(15, 26))
    assert not stem_loop.contains_near("chr1", 14, dist=1)
    assert not stem_loop.contains_near("chr1", 26, dist=1)


def test_contains_near_requires_matching_contig_stack_and_hairpin():
    complete_stem_loop = make_stem_loop(
        stacks=[StemLoopPart(bounds=[(16, 19)], energy=-1.0, desc="STACK:AAA/TTT")],
        hairpin=StemLoopPart(bounds=[(20, 21)], energy=2.0, desc="HAIRPIN:A/T"),
    )
    no_hairpin = make_stem_loop(stacks=complete_stem_loop.stacks)
    no_stack = make_stem_loop(hairpin=complete_stem_loop.hairpin)

    assert not complete_stem_loop.contains_near("chr2", 17, dist=1)
    assert not no_hairpin.contains_near("chr1", 17, dist=1)
    assert not no_stack.contains_near("chr1", 17, dist=1)


def test_find_near_stem_loop_selects_lowest_energy_candidate(monkeypatch):
    position = 50
    higher_energy_loop = StemLoop(
        stacks=[StemLoopPart(bounds=[(position, position)], energy=-0.5, desc="STACK:A/T")],
        hairpin=StemLoopPart(bounds=[(position, position)], energy=1.5, desc="HAIRPIN:A/T"),
        contig="chr1",
        start_position=45,
        sequence="A" * 25,
    )
    lower_energy_loop = StemLoop(
        stacks=[StemLoopPart(bounds=[(position, position)], energy=-1.5, desc="STACK:A/T")],
        hairpin=StemLoopPart(bounds=[(position, position)], energy=1.5, desc="HAIRPIN:A/T"),
        contig="chr1",
        start_position=30,
        sequence="A" * 25,
    )

    class Reference:
        def get_reference_length(self, contig):
            return 100

        def fetch(self, contig, start, end):
            return "A" * (end - start)

    def fold_window(sequence, contig, start_position):
        return [higher_energy_loop] if start_position == 45 else [lower_energy_loop]

    monkeypatch.setattr(sequence_utils, "fold_sequence", fold_window)
    result = find_near_stem_loop(Reference(), "chr1", position)

    assert result == (lower_energy_loop, 0)


def test_sequences_fold_into_stem_loops():
    start_position = 52090316

    sequence_with_initial_c = "TTATCTCCTAAGCTATGGCTTAGATT"
    stem_loops_with_initial_c = fold_sequence(sequence_with_initial_c, contig="chr20", start_position=start_position)

    assert len(stem_loops_with_initial_c) == 1
    assert stem_loops_with_initial_c[0].stacks
    assert stem_loops_with_initial_c[0].hairpin is not None
    assert stem_loops_with_initial_c[0].stacks[0].desc == "STACK:CTAAGC/GCTTAG"
    initial_c_position = start_position + sequence_with_initial_c.index("C")
    assert stem_loops_with_initial_c[0].contains_near("chr20", initial_c_position)
    assert stem_loops_with_initial_c[0].contains_near("chr20", initial_c_position, dist=3)
    assert not stem_loops_with_initial_c[0].contains_near("chr20", initial_c_position, dist=2)

    sequence_with_initial_a = "TGGCATCCCTCGCTGGCAGGATGGAG"
    stem_loops_with_initial_a = fold_sequence(sequence_with_initial_a, contig="chr20", start_position=start_position)
    initial_a_position = start_position + sequence_with_initial_a.index("A")

    assert stem_loops_with_initial_a
    assert any(stem_loop.contains_near("chr20", initial_a_position) for stem_loop in stem_loops_with_initial_a)

    sequence_with_final_g = "AGAGATGAGGCCTGCAAACGGCATCT"
    stem_loops_with_final_g = fold_sequence(sequence_with_final_g, contig="chr20", start_position=start_position)
    final_g_position = start_position + len(sequence_with_final_g) - 6

    assert sequence_with_final_g[-6] == "G"
    assert len(stem_loops_with_final_g[0].stacks) == 1
    assert stem_loops_with_final_g[0].stacks[0].bounds == [
        (start_position + 6, start_position + 10),
        (start_position + 20, start_position + 24),
    ]
    assert contains_near_stem_loop(stem_loops_with_final_g, "chr20", final_g_position)
    assert not contains_near_stem_loop(stem_loops_with_final_g, "chr20", final_g_position, max_energy=0.5)
