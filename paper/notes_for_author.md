# Notes for the author

Remove this file before the manuscript goes anywhere. It lists what in `manuscript.md` needs your judgement.

## Status

The manuscript, its figures and the code behind them were produced with Claude (Anthropic) on 27 November 2026, in one working session. Nothing has been checked by a second person. Every number in the text comes from the result files in `results/`; `python3 paper/quoted_numbers.py` prints them all, so each one can be traced.

Most journals ask for a statement on the use of AI tools. That statement is not in the draft yet.

## Decisions that are yours

1. **Authors and affiliation.** Only your name is on the draft, with placeholders for co-authors, affiliation, e-mail, acknowledgements and funding. Ángel Goñi-Moreno is cited but not listed.
2. **Title.** The current one mirrors the 2020 title. Alternatives: "Co-translational dimerisation revisited: the exposure of the interface, not the overlap of chains, sets programmable nonlinearity"; "A competence-window model of co-translational dimerisation".
3. **How hard to be on the 2020 paper.** The draft says the core idea holds and the mechanism does not. Two further points about the 2020 equations are *not* in the draft:
   - Equation 3 of the 2020 paper is a probability per pair of ribosomes, whereas equations 4 and 5 use α as a fraction of expression. Each chain has two neighbours, so at low α the fraction of chains is about twice the pair probability. The new equation 4 handles this through the factor 2q/(1+q).
   - Equations 4 and 5 of the 2020 paper turn two monomers into one dimer without a factor 2.
4. **The toggle switch.** The draft only says that the toggle switch "lost bistability in a similar manner", citing the 2020 paper. It was not re-analysed here. In `ToggleSwitch.nb`, the local-dimer terms in `eqs2` have inconsistent indices (the term `αloc·Ρ1l·A1` appears in the equations of both A1 and A2, and Ρ2l in neither). With consistent indices the proximity effect of the 2020 Figure 4 disappears in my re-implementation. This needs your check before the toggle switch is mentioned at all.

## Claims to verify

5. **Brandt et al. 2009 (reference 12).** I could only read the abstract; the publisher blocks automated access. The sentence in section 2.2 paraphrases that abstract. The geometry in the paper comes from PDB 9GFT (two collided ribosomes) and from randomly oriented pairs, not from the tomographic arrangements of Brandt et al. Read the paper and check the paraphrase. Full reference: Brandt F, Etchells SA, Ortiz JO, Elcock AH, Hartl FU, Baumeister W. The native 3D organization of bacterial polysomes. *Cell* 136(2):261–271 (2009). doi:10.1016/j.cell.2008.11.016
6. **Same RNA or not, in *E. coli*.** The *E. coli* data of Bertolini et al. come from T7 overexpression. The draft states that DiSP cannot tell whether the two ribosomes of a pair translate the same RNA under those conditions. This is the weakest link between the data and the model.
7. **The three repressors.** "α = 0 for TetR, LacI and λ cI" holds only if nearly the whole interface must be outside the tunnel. If half is enough, α is 0.01–0.07, 0.41–0.51 and 0.50–0.60. Section 2.5 presents both.
8. **The 3′ end of the profile.** No model reproduces the measured profile over the last 50 codons (models drop to about 0.5, the data stay at 0.6). The draft says so and offers slow termination as a possible reason without evidence.
9. **Assumptions behind the effective concentration.** RNA between ribosomes as a flexible chain (5.9 Å per nucleotide, Kuhn length 20 Å); nascent chains as unfolded coils; a 30-residue tunnel; pairing is irreversible. A stiffer RNA lowers the effective concentration at large distances.
10. **Elongation rate.** The new simulations use 15 residues per second (Young & Bremer); the 2020 paper used 10. Results are stated in terms of the window W, so this only rescales codon numbers.

## References

11. Reference numbers follow the order of first citation. Kohn et al., Natan et al. and Lewis et al. are shortened with "et al." in the list.
12. The TetR structure (PDB 4AC0) has no associated paper; it is cited as a database entry. A published TetR dimer structure may be preferable.
13. Bibliographic details were checked against Crossref, Europe PMC and the PDB. The descriptions of what each paper shows were written from memory, except for Bertolini et al. (full text read) and Brandt et al. (abstract read).

## Style

The wording follows the 2020 preprint: first person plural, British spelling ("dimerisation", "modelling"), connectors such as "This is," and "In what follows", bracketed roman numerals for lists, "in-vivo" with a hyphen, and the same section order (Introduction, Results and Discussion, Conclusion, Methods).

## Regenerating

    python3 compute_geometry.py && python3 compute_kernel.py     # structure-based geometry (about 10 minutes)
    python3 compute_tests.py && python3 compute_model.py         # tests against data, validation of the model
    python3 paper/make_figures.py                                # figures 1 to 5
    python3 paper/quoted_numbers.py                              # every number quoted in the text

The manuscript is plain Markdown with LaTeX mathematics, which GitHub renders. `pandoc paper/manuscript.md -o manuscript.docx` (or `.tex`) converts it for submission.
