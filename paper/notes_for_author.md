# Notes for the author

Remove this file before the manuscript goes anywhere. It lists what in `manuscript.md` needs your judgement.

## Status

The manuscript, its figures and the code behind them were produced with Claude (Anthropic) on 27–28 November 2026, in one working session. Nothing has been checked by a second person. Every number in the text comes from the result files in `results/`; `python3 paper/quoted_numbers.py` prints them all, so each one can be traced.

Most journals ask for a statement on the use of AI tools. A draft is in the section "Declaration of AI use"; the bracketed part, on what you checked or rewrote, is yours to complete.

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

## The three-dimensional simulation (section 2.4, Figure 4, Movie 1)

11. **What is modelled.** Chains of beads (4 residues per bead) on two rigid ribosomes with the shape of PDB 9GFT (without HrpA), moved by Metropolis Monte Carlo with diffusion coefficients from the Stokes–Einstein relation in water. Not included: hydrodynamic interactions, exclusion between the two chains, folding of mCherry, other ribosomes on the RNA, and the RNA outside the pair. The RNA between the two ribosomes is a spring; the path drawn in the pictures is schematic.
12. **Pairing on first contact.** Every contact between in-register interface beads of two competent chains leads to irreversible pairing. The time-lapse follows only microseconds of motion per 0.27 s of translation; this compression, not chemistry, sets when the chains pair in Figure 4G and in the movie. The text says so, and reads the compression as an effective pairing rate. After pairing, the paired stretch gets the length and stiffness of a coiled coil; that part is cosmetic.
13. **The cell.** Section 2.4 uses the translational diffusion of polysomes in *E. coli* (0.04 µm² s⁻¹, Bakshi et al. 2012, abstract read) as a worst case for how fast two neighbouring ribosomes move relative to each other. The bound "the encounter cannot slow down more than in proportion to the mobility" is my argument (monotonicity of the slowest relaxation rate in the diffusion coefficients), not a cited result.
14. **Rendering.** Figure 2A, Figure 4A–F and Movie 1 were rendered with the NGL viewer in headless Chrome (`paper/render/`). The rendered images are stored, so `make_figures.py` does not need Chrome. Journals usually want movies as MP4 (`figures/movie.mp4`); the GIF in the manuscript is a preview for GitHub.

## References

15. Reference numbers follow the order of first citation. Kohn et al., Natan et al. and Lewis et al. are shortened with "et al." in the list.
16. The TetR structure (PDB 4AC0) has no associated paper; it is cited as a database entry. A published TetR dimer structure may be preferable.
17. Bibliographic details were checked against Crossref, Europe PMC and the PDB. The descriptions of what each paper shows were written from memory, except for Bertolini et al. (full text read), and Brandt et al. and Bakshi et al. (abstracts read).

## Style

The wording follows the 2020 preprint: first person plural, British spelling ("dimerisation", "modelling"), connectors such as "This is," and "In what follows", bracketed roman numerals for lists, "in-vivo" with a hyphen, and the same section order (Introduction, Results and Discussion, Conclusion, Methods).

## Regenerating

    python3 compute_geometry.py && python3 compute_kernel.py     # structure-based geometry (about 10 minutes)
    python3 compute_tests.py && python3 compute_model.py         # tests against data, validation of the model
    python3 compute_3d.py                                        # three-dimensional simulations (about 90 minutes)
    python3 paper/render/render.py && python3 paper/render/movie.py   # molecular graphics (needs Chrome and ffmpeg)
    python3 paper/make_figures.py                                # figures 1 to 6
    python3 paper/quoted_numbers.py                              # every number quoted in the text

The manuscript is plain Markdown with LaTeX mathematics, which GitHub renders. `pandoc paper/manuscript.md -o manuscript.docx` (or `.tex`) converts it for submission.
