from typing import NamedTuple

import seqfold


class StemLoopPart(NamedTuple):
    bounds: list[tuple[int, int]]
    energy: float
    desc: str


class StemLoop(NamedTuple):
    stacks: list[StemLoopPart]
    hairpin: StemLoopPart | None
    contig: str
    start_position: int
    sequence: str

    @property
    def total_energy(self) -> float:
        stack_energy = sum(stack.energy for stack in self.stacks)
        hairpin_energy = 0.0 if self.hairpin is None else self.hairpin.energy
        return stack_energy + hairpin_energy

    def contains_near(self, contig: str, position: int, dist: int = 3) -> bool:
        return (
            contig == self.contig
            and bool(self.stacks)
            and self.hairpin is not None
            and any(
                start - dist <= position <= end + dist
                for stack in self.stacks
                for start, end in stack.bounds
            )
        )


def consolidate_adjacent_stacks(structures, contig: str, start_position: int, sequence: str):
    stem_loops = []
    pending_stacks = []
    stack_indices = []
    stack_energy = 0.0
    stack_top_bases = []
    stack_bottom_bases = []

    def flush_stack():
        nonlocal stack_energy
        if stack_indices:
            pending_stacks.append(
                StemLoopPart(
                    bounds=[
                        (start_position + stack_indices[0][0], start_position + stack_indices[-1][0]),
                        (start_position + stack_indices[-1][1], start_position + stack_indices[0][1]),
                    ],
                    energy=stack_energy,
                    desc="STACK:" + "".join(stack_top_bases) + "/" + "".join(stack_bottom_bases)[::-1],
                )
            )
            stack_indices.clear()
            stack_energy = 0.0
            stack_top_bases.clear()
            stack_bottom_bases.clear()

    for structure in structures:
        is_stack = structure.desc.startswith("STACK:") and structure.ij
        is_hairpin = structure.desc.startswith("HAIRPIN:") and structure.ij
        if not is_stack and not is_hairpin:
            continue

        is_adjacent = (
            stack_indices
            and structure.ij[0][0] == stack_indices[-1][0] + 1
            and structure.ij[0][1] == stack_indices[-1][1] - 1
        )
        if not is_stack or (stack_indices and not is_adjacent):
            flush_stack()
        if is_stack:
            stack_indices.extend(structure.ij)
            stack_energy += structure.e
            top_bases, bottom_bases = structure.desc.removeprefix("STACK:").split("/")
            stack_top_bases.append(top_bases if not stack_top_bases else top_bases[1:])
            stack_bottom_bases.append(bottom_bases if not stack_bottom_bases else bottom_bases[1:])
        else:  # hairpin
            stem_loops.append(
                StemLoop(
                    stacks=pending_stacks.copy(),
                    hairpin=StemLoopPart(
                        bounds=[
                            (start_position + structure.ij[0][0], start_position + structure.ij[0][1])
                        ],
                        energy=structure.e,
                        desc=structure.desc,
                    ),
                    contig=contig,
                    start_position=start_position,
                    sequence=sequence,
                )
            )
            pending_stacks.clear()

    flush_stack()
    if pending_stacks:
        stem_loops.append(
            StemLoop(
                stacks=pending_stacks,
                hairpin=None,
                contig=contig,
                start_position=start_position,
                sequence=sequence,
            )
        )
    return stem_loops


def fold_sequence(sequence: str, contig: str, start_position: int) -> list[StemLoop]:
    structures = seqfold.fold(sequence, temp=70)
    return consolidate_adjacent_stacks(structures, contig, start_position, sequence)


def contains_near_stem_loop(
    stem_loops: list[StemLoop], contig: str, position: int, dist: int = 3, max_energy: float = 1
) -> bool:
    return any(
        stem_loop.contains_near(contig, position, dist=dist) and stem_loop.total_energy < max_energy
        for stem_loop in stem_loops
    )


def is_near_stem_loop(reference, contig: str, position: int, dist: int = 3, max_energy: float = 1) -> bool:
    contig_length = reference.get_reference_length(contig)
    if not 0 <= position < contig_length:
        return False

    upstream_start = max(0, position - 5)
    upstream_end = min(contig_length, position + 20)
    upstream_sequence = reference.fetch(contig, upstream_start, upstream_end)
    downstream_start = max(0, position - 20)
    downstream_end = min(contig_length, position + 5)
    downstream_sequence = reference.fetch(contig, downstream_start, downstream_end)

    upstream_structure = fold_sequence(upstream_sequence, contig, upstream_start)
    downstream_structure = fold_sequence(downstream_sequence, contig, downstream_start)
    return contains_near_stem_loop(
        upstream_structure + downstream_structure, contig, position, dist=dist, max_energy=max_energy
    )