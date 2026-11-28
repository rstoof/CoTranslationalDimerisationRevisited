/*
 * Second-generation stochastic simulations (exact Gillespie) for testing the theory against data.
 *
 *   simulate_disp   one mRNA in steady state; reports what Disome Selective Profiling would
 *                   measure: per codon, the fraction of ribosomes whose nascent chain is bound to
 *                   the nascent chain of another ribosome that is still on the mRNA.
 *
 *   simulate_locus  one gene with coupled transcription and translation: RNA polymerases start
 *                   at the promoter, the lead ribosome follows the polymerase, further ribosomes
 *                   load on the nascent transcript, finished transcripts stay near the gene for
 *                   their lifetime. Nascent chains can pair within a polysome (cis), between
 *                   polysomes on sibling transcripts still attached to the DNA, and between
 *                   transcripts sharing the gene's territory.
 *
 * Pairing kernels (kern[0] selects):
 *   0  paper: hazard = kc * Psi^2, overlap of spheres with contour-length radii on a straight mRNA
 *   2  structure-based: hazard = kon * c_eff, with c_eff from a table computed by Monte Carlo
 *      around the atomic structure of two E. coli ribosomes (compute_kernel.py)
 *   3  constant: hazard = k for every eligible pair closer than s_max codons
 * All kernels can be gated: both chains must expose at least n_on residues (the interface).
 *
 * Build: cc -O2 -shared -fPIC -o libpolysome2.so polysome2.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAXR 400
#define MAXT 400
#define MAXRT 96
#define NA_A3 6.02214e-4 /* molecules per A^3 at 1 mol/l */

/* Random numbers: xoshiro256** with splitmix64 seeding, after the public-domain reference code of
 * D. Blackman and S. Vigna (https://prng.di.unimi.it/; ACM Trans. Math. Softw. 47:36, 2021). */
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

/* ----------------------------------------------------------------------------------- kernels
 * kern[0] mode      kern[1] kc (1/s)     kern[2] x_aa (A)      kern[3] x_codon (A)
 * kern[4] kon (1/M/s)   kern[5..7] unused
 * kern[8] linker0 (nt)  kern[9] nt_len (A)  kern[10] b_rna (A)  kern[11] s_min (codons)
 * kern[12] n_on   kern[13] tunnel   kern[14] k_const (1/s)   kern[15] s_max (codons)
 * kern[16] rg_prefactor (A)  kern[17] rg_exponent
 */
static double psi_spheres(double r1, double r2, double d) {
    double rs = r1 < r2 ? r1 : r2, rl = r1 < r2 ? r2 : r1;
    if (rs <= 0.0 || d >= r1 + r2) return 0.0;
    if (d <= rl - rs) return 1.0;
    double a = r1 + r2 - d;
    return a * a * (d * d + 2.0 * d * (r1 + r2) - 3.0 * (r1 - r2) * (r1 - r2)) / (16.0 * d * rs * rs * rs);
}

/* Effective concentration (mol/l) from the structure-based table:
 * axis 0 exposed residues of the chain on the leading ribosome, axis 1 of the trailing ribosome,
 * axis 2 codons between the ribosomes beyond the collided spacing. */
#define TMAX 24
static int TN = 0, TE = 0;
static double tab_n[TMAX], tab_e[TMAX], tab_c[TMAX * TMAX * TMAX];

void set_table(const double *n, int nn, const double *e, int ne, const double *log10c) {
    TN = nn; TE = ne;
    memcpy(tab_n, n, sizeof(double) * nn); memcpy(tab_e, e, sizeof(double) * ne);
    memcpy(tab_c, log10c, sizeof(double) * nn * nn * ne);
}

static void locate(const double *ax, int m, double x, int *i, double *w) {
    if (x <= ax[0]) { *i = 0; *w = 0.0; return; }
    if (x >= ax[m - 1]) { *i = m - 2; *w = 1.0; return; }
    int k = 0; while (x > ax[k + 1]) k++;
    *i = k; *w = (x - ax[k]) / (ax[k + 1] - ax[k]);
}

double ceff_table(double na, double nb, double extra) {
    if (TN < 2 || na < tab_n[0] || nb < tab_n[0]) return 0.0; /* too little chain outside the tunnel */
    double dil = 1.0;
    if (extra > tab_e[TE - 1]) { dil = pow(tab_e[TE - 1] / extra, 1.5); extra = tab_e[TE - 1]; } /* tether dilution */
    if (extra < 0) extra = 0;
    int ia, ib, ie; double wa, wb, we;
    locate(tab_n, TN, na, &ia, &wa); locate(tab_n, TN, nb, &ib, &wb); locate(tab_e, TE, extra, &ie, &we);
    double v = 0.0;
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int e = 0; e < 2; e++)
        v += (a ? wa : 1 - wa) * (b ? wb : 1 - wb) * (e ? we : 1 - we) * tab_c[((ia + a) * TN + (ib + b)) * TE + (ie + e)];
    return dil * pow(10.0, v);
}

/* hazard for two chains on the same mRNA: exposed lengths ni >= nj are not assumed; sep in codons */
static double hazard_cis(const double *kern, int pi, int pj) {
    int tunnel = (int)kern[13], n_on = (int)kern[12];
    int ni = pi - tunnel, nj = pj - tunnel, sep = pi > pj ? pi - pj : pj - pi;
    if (ni < n_on || nj < n_on || ni <= 0 || nj <= 0) return 0.0;
    int mode = (int)kern[0];
    if (mode == 0) {
        double p = psi_spheres(kern[2] * ni, kern[2] * nj, kern[3] * sep);
        return kern[1] * p * p;
    } else if (mode == 2) {
        double extra = sep - kern[11]; if (extra < 0) extra = 0;
        return kern[4] * (pi > pj ? ceff_table(ni, nj, extra) : ceff_table(nj, ni, extra));
    } else {
        return sep <= (int)kern[15] ? kern[14] : 0.0;
    }
}

/* ------------------------------------------------------------------------- single-mRNA DiSP
 * par[0] L   par[1] lam   par[2] footprint   par[3] v (codons/s)   par[4] kterm (1/s)
 * par[5] T   par[6] T_burn   par[7] seed   par[8] neighbours_only
 * out[0] chains released in a co-translational dimer   out[1] released free
 * out[2] initiations   out[3] ribosome-seconds   out[4] measured time   out[5] pairing events
 * occ_all[0..L], occ_link[0..L]: ribosome-seconds per codon, total and linked
 * hist_sep[0..L]: codon separation at pairing;  hist_pos[0..L]: position of the trailing ribosome at pairing
 */
int simulate_disp(const double *par, const double *kern, double *out,
                  double *occ_all, double *occ_link, double *hist_sep, double *hist_pos) {
    int L = (int)par[0];
    double lam = par[1];
    int fp = (int)par[2];
    double v = par[3], kterm = par[4], T = par[5], Tburn = par[6];
    seed_rng((uint64_t)par[7]);
    int nb_only = (int)par[8];
    static int pos[MAXR], st[MAXR], partner[MAXR];
    static double hop[MAXR], haz[MAXR * 16];
    static int hi[MAXR * 16], hj[MAXR * 16];
    int n = 0;
    double t = 0.0;
    for (int i = 0; i < 6; i++) out[i] = 0.0;
    memset(occ_all, 0, sizeof(double) * (L + 1)); memset(occ_link, 0, sizeof(double) * (L + 1));
    memset(hist_sep, 0, sizeof(double) * (L + 1)); memset(hist_pos, 0, sizeof(double) * (L + 1));

    while (t < T) {
        double total = 0.0;
        double r_init = (n < MAXR - 1 && (n == 0 || pos[n - 1] >= fp)) ? lam : 0.0;
        total += r_init;
        for (int i = 0; i < n; i++) {
            if (pos[i] >= L) hop[i] = (i == 0) ? kterm : 0.0;
            else if (i == 0 || pos[i - 1] - pos[i] > fp) hop[i] = v;
            else hop[i] = 0.0;
            total += hop[i];
        }
        int nh = 0;
        for (int i = 0; i < n && nh < MAXR * 16 - 2; i++) {
            if (st[i]) continue;
            for (int j = i + 1; j < n && nh < MAXR * 16 - 2; j++) {
                if (st[j]) { if (nb_only) break; else continue; }
                double h = hazard_cis(kern, pos[i], pos[j]);
                if (h > 0.0) { haz[nh] = h; hi[nh] = i; hj[nh] = j; nh++; total += h; }
                if (nb_only) break;
            }
        }
        double dt = -log(urand()) / total, tn = t + dt;
        if (tn > Tburn) {
            double w = tn - (t > Tburn ? t : Tburn);
            out[3] += w * n; out[4] += w;
            for (int i = 0; i < n; i++) { occ_all[pos[i]] += w; if (st[i] == 1) occ_link[pos[i]] += w; }
        }
        t = tn;
        if (t >= T) break;
        double u = urand() * total;
        if (u < r_init) { pos[n] = 0; st[n] = 0; partner[n] = -1; n++; if (t > Tburn) out[2] += 1; continue; }
        u -= r_init;
        int done = 0;
        for (int i = 0; i < n; i++) {
            if (u < hop[i]) {
                if (pos[i] >= L) { /* termination of the lead ribosome (index 0) */
                    if (t > Tburn) out[st[0] ? 0 : 1] += 1;
                    for (int k = 1; k < n; k++) {
                        pos[k - 1] = pos[k]; st[k - 1] = st[k];
                        int p = partner[k];
                        if (p == 0) { st[k - 1] = 2; partner[k - 1] = -1; } /* partner left: back to monosome */
                        else partner[k - 1] = p > 0 ? p - 1 : -1;
                    }
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
                st[i] = 1; st[j] = 1; partner[i] = j; partner[j] = i;
                if (t > Tburn) { out[5] += 1; hist_sep[pos[i] - pos[j]] += 1; hist_pos[pos[j]] += 1; }
                break;
            }
            u -= haz[k];
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------------- gene locus
 * par[0] L (codons)        par[1] k_tx (1/s, when promoter on)   par[2] v_tx (codons/s)
 * par[3] lam (1/s)         par[4] v_rib (codons/s)               par[5] kterm (1/s)
 * par[6] footprint         par[7] rnap_footprint (codons)        par[8] rnap_gap (codons)
 * par[9] mrna_life (s)     par[10] T                             par[11] T_burn
 * par[12] seed             par[13] p_on rate (1/s; <=0: always on)  par[14] p_off rate (1/s)
 * par[15] territory volume (litre; <=0: no territory pairing)    par[16] bp_len (A per bp of DNA)
 * par[17] unused (a ribosome can never pass the polymerase; T7 is modelled by a faster v_tx)
 * par[18] refresh (recompute pair hazards every this many events)
 *
 * out[0] chains released free            out[1] paired in cis (same mRNA)
 * out[2] paired between sibling nascent transcripts   out[3] paired within the territory
 * out[4] measured time    out[5] transcript-seconds   out[6] nascent-transcript-seconds
 * out[7] ribosome-seconds out[8] promoter-on seconds  out[9] transcripts made
 * out[10] chains released from transcripts still attached to the DNA
 */
typedef struct {
    int used, nascent, dead, rnap, nrib;
    double born;
    int pos[MAXRT], st[MAXRT];
} Tr;

int simulate_locus(const double *par, const double *kern, double *out) {
    int L = (int)par[0];
    double ktx = par[1], vtx = par[2], lam = par[3], vr = par[4], kterm = par[5];
    int fp = (int)par[6], rfp = (int)par[7], rgap = (int)par[8];
    double life = par[9], T = par[10], Tburn = par[11];
    seed_rng((uint64_t)par[12]);
    double pon = par[13], poff = par[14], vterr = par[15], bp_len = par[16];
    int refresh = (int)par[18];
    if (refresh < 1) refresh = 1;
    int tunnel = (int)kern[13], n_on = (int)kern[12];
    double kon = kern[4];
    double h_terr = vterr > 0 ? kon / (6.02214e23 * vterr) : 0.0;

    static Tr tr[MAXT];
    static double ph[40000]; static int pa[40000], pb[40000], pc[40000], pd[40000], pk[40000];
    memset(tr, 0, sizeof(tr));
    for (int i = 0; i < 12; i++) out[i] = 0.0;
    int promoter = 1, np = 0, stale = 1, since = 0;
    double t = 0.0, hsum = 0.0, terr_pairs = 0.0;
    int tail = 10; /* codons of 3' untranslated region before release */

    while (t < T) {
        /* ---- pair hazards (cached between refreshes) */
        if (stale || since >= refresh) {
            np = 0; hsum = 0.0; since = 0; stale = 0;
            int nel = 0, nel_nasc = 0; double same = 0.0;
            for (int a = 0; a < MAXT; a++) {
                if (!tr[a].used) continue;
                int ea = 0;
                for (int i = 0; i < tr[a].nrib; i++) {
                    if (tr[a].st[i] || tr[a].pos[i] - tunnel < n_on) continue;
                    ea++;
                    for (int j = i + 1; j < tr[a].nrib && np < 39990; j++) { /* cis */
                        if (tr[a].st[j]) continue;
                        double h = hazard_cis(kern, tr[a].pos[i], tr[a].pos[j]);
                        if (h > 0) { ph[np] = h; pa[np] = a; pb[np] = i; pc[np] = a; pd[np] = j; pk[np] = 1; np++; hsum += h; }
                    }
                    if (!tr[a].nascent) continue;
                    for (int b = a + 1; b < MAXT && np < 39990; b++) { /* sibling nascent transcripts */
                        if (!tr[b].used || !tr[b].nascent) continue;
                        double dz = bp_len * 3.0 * fabs((double)(tr[a].rnap - tr[b].rnap));
                        for (int j = 0; j < tr[b].nrib && np < 39990; j++) {
                            if (tr[b].st[j] || tr[b].pos[j] - tunnel < n_on) continue;
                            double la = 3.0 * (tr[a].rnap - tr[a].pos[i]), lb = 3.0 * (tr[b].rnap - tr[b].pos[j]);
                            if (la < 0) la = 0; if (lb < 0) lb = 0;
                            /* equivalent flexible tether: both mRNA stretches plus the DNA between the polymerases */
                            double r2 = (la + lb) * kern[9] * kern[10] + dz * dz;
                            double extra = (r2 / (kern[9] * kern[10]) - kern[8]) / 3.0; if (extra < 2.0) extra = 2.0;
                            double n1 = tr[a].pos[i] - tunnel, n2 = tr[b].pos[j] - tunnel;
                            double h = kon * (n1 > n2 ? ceff_table(n1, n2, extra) : ceff_table(n2, n1, extra));
                            if (h > 0) { ph[np] = h; pa[np] = a; pb[np] = i; pc[np] = b; pd[np] = j; pk[np] = 2; np++; hsum += h; }
                        }
                    }
                }
                nel += ea; same += 0.5 * ea * (ea - 1.0);
                if (tr[a].nascent) nel_nasc += ea;
            }
            /* territory pairs: different transcripts, not both nascent */
            double nascpairs = 0.0;
            if (h_terr > 0) {
                double sn = 0.0; /* same-transcript pairs among nascent */
                for (int a = 0; a < MAXT; a++) if (tr[a].used && tr[a].nascent) {
                    int ea = 0; for (int i = 0; i < tr[a].nrib; i++) if (!tr[a].st[i] && tr[a].pos[i] - tunnel >= n_on) ea++;
                    sn += 0.5 * ea * (ea - 1.0);
                }
                nascpairs = 0.5 * nel_nasc * (nel_nasc - 1.0) - sn;
                terr_pairs = 0.5 * nel * (nel - 1.0) - same - nascpairs;
                if (terr_pairs < 0) terr_pairs = 0;
            } else terr_pairs = 0.0;
        }
        /* ---- single-entity rates */
        double total = hsum + h_terr * terr_pairs;
        double r_sw = 0.0;
        if (pon > 0) r_sw = promoter ? poff : pon;
        total += r_sw;
        int slot = -1, clear = 1, ntr = 0, nnasc = 0, nrib = 0;
        for (int a = 0; a < MAXT; a++) {
            if (!tr[a].used) { if (slot < 0) slot = a; continue; }
            ntr++; nrib += tr[a].nrib;
            if (tr[a].nascent) { nnasc++; if (tr[a].rnap < rfp) clear = 0; }
        }
        double r_tx = (promoter && clear && slot >= 0) ? ktx : 0.0;
        total += r_tx;
        double r_ent = 0.0; /* polymerase steps, mRNA inactivation, ribosome initiation, hops */
        for (int a = 0; a < MAXT; a++) {
            if (!tr[a].used) continue;
            if (tr[a].nascent) r_ent += vtx;
            if (!tr[a].dead) { r_ent += 1.0 / life; r_ent += lam; }
            r_ent += tr[a].nrib * vr;
        }
        total += r_ent; /* upper bound; blocked moves are rejected below (null events) */
        double dt = -log(urand()) / total, tn = t + dt;
        if (tn > Tburn) {
            double w = tn - (t > Tburn ? t : Tburn);
            out[4] += w; out[5] += w * ntr; out[6] += w * nnasc; out[7] += w * nrib; out[8] += w * promoter;
        }
        t = tn; since++;
        if (t >= T) break;
        double u = urand() * total;
        /* pairing */
        if (u < hsum) {
            for (int k = 0; k < np; k++) {
                if (u < ph[k] || k == np - 1) {
                    Tr *A = &tr[pa[k]], *B = &tr[pc[k]];
                    if (A->used && B->used && pb[k] < A->nrib && pd[k] < B->nrib && !A->st[pb[k]] && !B->st[pd[k]]) {
                        A->st[pb[k]] = pk[k]; B->st[pd[k]] = pk[k]; stale = 1;
                    }
                    break;
                }
                u -= ph[k];
            }
            continue;
        }
        u -= hsum;
        if (u < h_terr * terr_pairs) { /* pick a territory-class pair uniformly among eligible chains */
            int nel = 0;
            for (int a = 0; a < MAXT; a++) if (tr[a].used)
                for (int i = 0; i < tr[a].nrib; i++) if (!tr[a].st[i] && tr[a].pos[i] - tunnel >= n_on) nel++;
            for (int tries = 0; tries < 400 && nel >= 2; tries++) {
                int k1 = (int)(urand() * nel), k2 = (int)(urand() * nel);
                if (k1 == k2) continue;
                int a1 = -1, i1 = -1, a2 = -1, i2 = -1, c = 0;
                for (int a = 0; a < MAXT; a++) if (tr[a].used)
                    for (int i = 0; i < tr[a].nrib; i++) if (!tr[a].st[i] && tr[a].pos[i] - tunnel >= n_on) {
                        if (c == k1) { a1 = a; i1 = i; }
                        if (c == k2) { a2 = a; i2 = i; }
                        c++;
                    }
                if (a1 < 0 || a2 < 0 || a1 == a2 || (tr[a1].nascent && tr[a2].nascent)) continue;
                tr[a1].st[i1] = 3; tr[a2].st[i2] = 3; stale = 1; break;
            }
            continue;
        }
        u -= h_terr * terr_pairs;
        if (u < r_sw) { promoter = !promoter; continue; }
        u -= r_sw;
        if (u < r_tx) {
            Tr *A = &tr[slot]; memset(A, 0, sizeof(Tr)); A->used = 1; A->nascent = 1; A->rnap = 0; A->born = t;
            if (t > Tburn) out[9] += 1; stale = 1; continue;
        }
        u -= r_tx;
        for (int a = 0; a < MAXT; a++) {
            Tr *A = &tr[a];
            if (!A->used) continue;
            if (A->nascent) {
                if (u < vtx) { /* polymerase step, blocked by a polymerase ahead */
                    int ok = 1;
                    for (int b = 0; b < MAXT; b++) if (b != a && tr[b].used && tr[b].nascent && tr[b].rnap > A->rnap && tr[b].rnap - A->rnap <= rfp) ok = 0;
                    if (ok) { A->rnap++; if (A->rnap >= L + tail) { A->nascent = 0; stale = 1; } }
                    goto next;
                }
                u -= vtx;
            }
            if (!A->dead) {
                if (u < 1.0 / life) { A->dead = 1; goto next; }
                u -= 1.0 / life;
                if (u < lam) {
                    if (A->rnap >= fp + rgap && A->nrib < MAXRT - 1 && (A->nrib == 0 || A->pos[A->nrib - 1] >= fp)) {
                        A->pos[A->nrib] = 0; A->st[A->nrib] = 0; A->nrib++;
                    }
                    goto next;
                }
                u -= lam;
            }
            if (u < A->nrib * vr) {
                int i = (int)(u / vr); if (i >= A->nrib) i = A->nrib - 1;
                if (A->pos[i] >= L) { /* termination (rate folded into vr * kterm/vr acceptance) */
                    if (i == 0 && urand() < kterm / vr) {
                        if (t > Tburn) { out[A->st[0]] += 1; if (A->nascent) out[10] += 1; }
                        for (int k = 1; k < A->nrib; k++) { A->pos[k - 1] = A->pos[k]; A->st[k - 1] = A->st[k]; }
                        A->nrib--; stale = 1;
                    }
                } else {
                    int ok = (i == 0 || A->pos[i - 1] - A->pos[i] > fp);
                    if (ok && A->nascent && A->pos[i] + 1 > A->rnap - rgap) ok = 0; /* cannot pass the polymerase */
                    if (ok) {
                        A->pos[i]++;
                        if (A->pos[i] - tunnel == n_on) stale = 1;
                    }
                }
                goto next;
            }
            u -= A->nrib * vr;
        }
    next:
        for (int a = 0; a < MAXT; a++) /* remove finished transcripts */
            if (tr[a].used && !tr[a].nascent && tr[a].dead && tr[a].nrib == 0) { tr[a].used = 0; stale = 1; }
    }
    return 0;
}
