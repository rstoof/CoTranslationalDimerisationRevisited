/*
 * Stochastic molecular simulation of co-translational dimerisation on a polysome.
 *
 * Exact Gillespie (SSA) simulation of one mRNA:
 *   - ribosomes initiate as a Poisson process (rate lam) when the start site is clear
 *   - each ribosome hops codon by codon (TASEP with an extended footprint,
 *     codon-specific rates)
 *   - every pair of nascent chains that have not yet dimerised can dimerise with
 *     a hazard that depends on the instantaneous geometry
 *       mode 0 ("spheres", Stoof & Goni-Moreno 2020): hazard = kc * Psi^2, with Psi
 *              the relative overlap volume of two spheres whose radii are the
 *              contour lengths of the nascent chains and whose centres sit on a
 *              straight mRNA.
 *       mode 1 ("polymer"): hazard = kon * c_eff, with c_eff the effective molar
 *              concentration of one N-terminal domain around the other for two
 *              random-coil chains tethered to neighbouring ribosomes.
 *
 * Build: cc -O2 -shared -fPIC -o libpolysome.so polysome.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAXR 512

static uint64_t s[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t next_u64(void) {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
}
static void seed_rng(uint64_t x) {
    for (int i = 0; i < 4; i++) {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[i] = z ^ (z >> 31);
    }
}
static inline double urand(void) { return ((next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

/* Relative overlap volume of two spheres (radii r1, r2, centre distance d),
 * normalised by the volume of the smaller sphere. Equation 2 of the paper,
 * extended with the containment case. */
double psi_spheres(double r1, double r2, double d) {
    double rs = r1 < r2 ? r1 : r2, rl = r1 < r2 ? r2 : r1;
    if (rs <= 0.0) return 0.0;
    if (d >= r1 + r2) return 0.0;
    if (d <= rl - rs) return 1.0;
    double a = r1 + r2 - d;
    double v = M_PI * a * a * (d * d + 2.0 * d * (r1 + r2) - 3.0 * (r1 - r2) * (r1 - r2)) / (12.0 * d);
    return v / (4.0 / 3.0 * M_PI * rs * rs * rs);
}

/* Effective concentration (mol/l) of N-terminus 2 at N-terminus 1 for two
 * Gaussian coils of n1 and n2 exposed residues tethered dc Angstrom apart. */
double ceff_polymer(double n1, double n2, double dc) {
    double rg1 = 1.93 * pow(n1, 0.598), rg2 = 1.93 * pow(n2, 0.598);
    double s2 = 2.0 * (rg1 * rg1 + rg2 * rg2); /* per-axis variance of the relative vector */
    double p = pow(2.0 * M_PI * s2, -1.5) * exp(-dc * dc / (2.0 * s2)); /* per A^3 */
    return p / 6.02214e-4;
}

/*
 * par[0]  L          protein length (codons)
 * par[1]  lam        initiation rate (1/s)
 * par[2]  footprint  ribosome footprint (codons)
 * par[3]  mode       0 spheres, 1 polymer
 * par[4]  kc         mode 0: hazard prefactor (1/s);  mode 1: kon (1/M/s)
 * par[5]  tunnel     residues hidden in the exit tunnel
 * par[6]  xaa        contour length per residue (A)
 * par[7]  xcodon     mRNA length per codon (A)
 * par[8]  Drib       ribosome diameter (A)                [mode 1]
 * par[9]  b_rna      Kuhn length of the mRNA spacer (A)   [mode 1]
 * par[10] ndom       residues needed before the interface exists [mode 1]
 * par[11] T          simulated time (s)
 * par[12] seed
 * par[13] kterm      termination rate (1/s)
 * par[14] snap_dt    snapshot interval (s)
 * par[15] max_snap   number of snapshots
 * par[16] T_burn     burn-in time (s) excluded from statistics
 * par[17] neighbours_only  1: only consecutive ribosomes may dimerise
 *
 * rate[0..L-1]  elongation rate for the hop from codon i to i+1
 * out[0] chains released as part of a co-translational dimer
 * out[1] chains released as free monomers
 * out[2] initiations, out[3] ribosome-seconds, out[4] measured time
 * out[5] blocked ribosome-seconds, out[6] dimerisation events
 * out[7] sum of ribosome dwell times of released chains
 * hist_len[0..L]  length of the shorter chain at the moment of dimerisation
 * hist_sep[0..L]  codon separation at the moment of dimerisation
 * snap[max_snap*64*2]  (position, state) of up to 64 ribosomes per snapshot;
 *                      state: 0 free, 1 dimerised, -1 empty slot
 */
int simulate(const double *par, const double *rate, double *out,
             double *hist_len, double *hist_sep, double *snap) {
    int L = (int)par[0];
    double lam = par[1];
    int fp = (int)par[2], mode = (int)par[3];
    double kc = par[4];
    int tunnel = (int)par[5];
    double xaa = par[6], xcodon = par[7], Drib = par[8], b_rna = par[9];
    int ndom = (int)par[10];
    double T = par[11];
    seed_rng((uint64_t)par[12]);
    double kterm = par[13], snap_dt = par[14];
    int max_snap = (int)par[15];
    double Tburn = par[16];
    int nb_only = (int)par[17];

    static int pos[MAXR], dim[MAXR];
    static double born[MAXR], hop[MAXR];
    static double haz[MAXR * 8];
    static int hi[MAXR * 8], hj[MAXR * 8];
    int n = 0, nsnap = 0;
    double t = 0.0;
    for (int i = 0; i < 8; i++) out[i] = 0.0;
    memset(hist_len, 0, sizeof(double) * (L + 1));
    memset(hist_sep, 0, sizeof(double) * (L + 1));

    while (t < T) {
        double total = 0.0;
        double r_init = (n < MAXR - 1 && (n == 0 || pos[n - 1] >= fp)) ? lam : 0.0;
        total += r_init;
        int nblocked = 0;
        for (int i = 0; i < n; i++) {
            if (pos[i] >= L) hop[i] = kterm;
            else if (i == 0 || pos[i - 1] - pos[i] > fp || pos[i - 1] >= L) hop[i] = rate[pos[i]];
            else { hop[i] = 0.0; nblocked++; }
            total += hop[i];
        }
        int nh = 0;
        for (int i = 0; i < n && nh < MAXR * 8 - 8; i++) {
            if (dim[i]) continue;
            int ni = pos[i] - tunnel;
            if (ni <= 0) continue;
            for (int j = i + 1; j < n && nh < MAXR * 8 - 8; j++) {
                if (dim[j]) { if (nb_only) break; else continue; }
                int nj = pos[j] - tunnel;
                if (nj <= 0) break; /* ribosomes behind j are even shorter */
                double h;
                int sep = pos[i] - pos[j];
                if (mode == 0) {
                    double p = psi_spheres(xaa * ni, xaa * nj, xcodon * sep);
                    h = kc * p * p;
                } else {
                    if (ni < ndom || nj < ndom) h = 0.0;
                    else {
                        double dc = sqrt(Drib * Drib + sep * xcodon * b_rna);
                        h = kc * ceff_polymer(ni, nj, dc);
                    }
                }
                if (h > 0.0) { haz[nh] = h; hi[nh] = i; hj[nh] = j; nh++; total += h; }
                if (nb_only) break;
            }
        }
        double dt = -log(urand()) / total;
        double tn = t + dt;
        while (nsnap < max_snap && nsnap * snap_dt + Tburn <= tn && nsnap * snap_dt + Tburn >= t) {
            double *sp = snap + (size_t)nsnap * 128;
            for (int k = 0; k < 64; k++) {
                if (k < n) { sp[2 * k] = pos[k]; sp[2 * k + 1] = dim[k]; }
                else { sp[2 * k] = -1; sp[2 * k + 1] = -1; }
            }
            nsnap++;
        }
        if (tn > Tburn) {
            double w = tn - (t > Tburn ? t : Tburn);
            out[3] += w * n; out[4] += w; out[5] += w * nblocked;
        }
        t = tn;
        if (t >= T) break;
        double u = urand() * total;
        if (u < r_init) {
            pos[n] = 0; dim[n] = 0; born[n] = t; n++;
            if (t > Tburn) out[2] += 1;
            continue;
        }
        u -= r_init;
        int done = 0;
        for (int i = 0; i < n; i++) {
            if (u < hop[i]) {
                if (pos[i] >= L) { /* termination: release chain i (always the lead) */
                    if (t > Tburn) { out[dim[i] ? 0 : 1] += 1; out[7] += t - born[i]; }
                    for (int k = i; k < n - 1; k++) { pos[k] = pos[k + 1]; dim[k] = dim[k + 1]; born[k] = born[k + 1]; }
                    n--;
                } else pos[i]++;
                done = 1; break;
            }
            u -= hop[i];
        }
        if (done) continue;
        for (int k = 0; k < nh; k++) {
            if (u < haz[k] || k == nh - 1) {
                int i = hi[k], j = hj[k];
                dim[i] = 1; dim[j] = 1;
                if (t > Tburn) {
                    out[6] += 1;
                    int nj = pos[j] - tunnel; if (nj > L) nj = L;
                    hist_len[nj] += 1;
                    int sep = pos[i] - pos[j]; if (sep > L) sep = L;
                    hist_sep[sep] += 1;
                }
                break;
            }
            u -= haz[k];
        }
    }
    return nsnap;
}
