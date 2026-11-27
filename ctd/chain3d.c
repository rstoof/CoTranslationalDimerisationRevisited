/*
 * Three-dimensional dynamics of two nascent chains on a pair of ribosomes.
 *
 * Coarse-grained Metropolis dynamics with local moves (an approximation of Brownian motion):
 *   - both ribosomes are rigid bodies that translate and rotate; they are tethered by the RNA
 *     between them (Gaussian spring with a finite contour length) and may not overlap;
 *   - each nascent chain is a string of beads (one bead = 4 residues, bond length 15.2 A)
 *     attached to the tunnel exit of its ribosome; beads may not come closer than a given
 *     clearance to any atom of either ribosome (clearance fields on grids, one per ribosome,
 *     in the coordinate frame of that ribosome);
 *   - beads of the dimerisation interface pair in register (bead i of chain A with bead i of
 *     chain B) when they meet: the first pair needs both chains to be competent, further pairs
 *     zip up next to existing ones;
 *   - a paired stretch takes the rise of an alpha-helix along the chain and resists bending, a
 *     crude picture of a coiled coil.
 *
 * The caller keeps the state between calls (poses, coordinates, paired beads, random stream),
 * adds beads as translation proceeds, and removes a ribosome when it terminates.
 *
 * Build: cc -O2 -shared -fPIC -o libchain3d.so chain3d.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

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
static inline double sym(double a) { return (2.0 * urand() - 1.0) * a; }

typedef struct { const float *g; const double *o; double h; const int *n; } Grid;

static inline double clearance(const Grid *G, const double *p) {
    int i = (int)floor((p[0] - G->o[0]) / G->h), j = (int)floor((p[1] - G->o[1]) / G->h), k = (int)floor((p[2] - G->o[2]) / G->h);
    if (i < 0 || j < 0 || k < 0 || i >= G->n[0] || j >= G->n[1] || k >= G->n[2]) return 99.0;
    return G->g[((size_t)i * G->n[1] + j) * G->n[2] + k];
}
/* pose: R (9, row-major) then t (3): lab = R native + t */
static inline void to_lab(const double *P, const double *x, double *y) {
    for (int a = 0; a < 3; a++) y[a] = P[3 * a] * x[0] + P[3 * a + 1] * x[1] + P[3 * a + 2] * x[2] + P[9 + a];
}
static inline void to_native(const double *P, const double *x, double *y) {
    double d[3] = {x[0] - P[9], x[1] - P[10], x[2] - P[11]};
    for (int a = 0; a < 3; a++) y[a] = P[a] * d[0] + P[3 + a] * d[1] + P[6 + a] * d[2];
}
static inline double dist(const double *a, const double *b) {
    double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return sqrt(x * x + y * y + z * z);
}

typedef struct { Grid G1, G2; const double *P1, *P2; int rib1, rib2; } World;

static inline int is_free(const World *W, const double *p, double c) {
    double q[3];
    if (W->rib1) { to_native(W->P1, p, q); if (clearance(&W->G1, q) < c) return 0; }
    if (W->rib2) { to_native(W->P2, p, q); if (clearance(&W->G2, q) < c) return 0; }
    return 1;
}
/* cosine of the angle between the bonds a->b and b->c (1 for a straight chain) */
static inline double cosang(const double *a, const double *b, const double *c) {
    double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
    double nu = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), nv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (nu < 1e-9 || nv < 1e-9) return 1.0;
    return (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
}
static inline int mid_free(const World *W, const double *a, const double *b, double c) {
    double m[3] = {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])};
    return is_free(W, m, c);
}
/* candidate pose: a small random rotation about the centre of the body and a small random step */
static int propose(const double *pose, const double *centre_n, double dtr, double drot, double *P, double *shift2, double *angle2) {
    double u[3] = {sym(1.0), sym(1.0), sym(1.0)}, nu = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    double th = sym(drot), d[3] = {sym(dtr), sym(dtr), sym(dtr)};
    if (nu < 1e-6) return 0;
    u[0] /= nu; u[1] /= nu; u[2] /= nu;
    double cs = cos(th), sn = sin(th), dR[9], c0[3];
    dR[0] = cs + u[0] * u[0] * (1 - cs);        dR[1] = u[0] * u[1] * (1 - cs) - u[2] * sn; dR[2] = u[0] * u[2] * (1 - cs) + u[1] * sn;
    dR[3] = u[1] * u[0] * (1 - cs) + u[2] * sn; dR[4] = cs + u[1] * u[1] * (1 - cs);        dR[5] = u[1] * u[2] * (1 - cs) - u[0] * sn;
    dR[6] = u[2] * u[0] * (1 - cs) - u[1] * sn; dR[7] = u[2] * u[1] * (1 - cs) + u[0] * sn; dR[8] = cs + u[2] * u[2] * (1 - cs);
    for (int a = 0; a < 3; a++) for (int e = 0; e < 3; e++)
        P[3 * a + e] = dR[3 * a] * pose[e] + dR[3 * a + 1] * pose[3 + e] + dR[3 * a + 2] * pose[6 + e];
    to_lab(pose, centre_n, c0);
    for (int a = 0; a < 3; a++)
        P[9 + a] = c0[a] + d[a] - (P[3 * a] * centre_n[0] + P[3 * a + 1] * centre_n[1] + P[3 * a + 2] * centre_n[2]);
    *shift2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    *angle2 = th * th;
    return 1;
}

/*
 * par[0] bond length b        par[1] bond stiffness K (kT/A^2)   par[2] bead step (A)
 * par[3] bead clearance (A)   par[4] pair distance d0 (A)        par[5] capture radius (A)
 * par[6] first interface bead par[7] one past the last           par[8] gate (interface beads exposed)
 * par[9] probability of nucleation per sweep in contact          par[10] tether contour length (A)
 * par[11] tether stiffness (kT/A^2; energy = k r^2)              par[12] rigid translation step (A)
 * par[13] rigid rotation step (rad)   par[14] clash clearance (A)   par[15] allowed clash fraction
 * par[16] chain A attached to its exit   par[17] chain B attached   par[18] tether active
 * par[19] subsampling step for the clash test   par[20] seed   par[21] first sweep number
 * par[22] ribosome 1 present   par[23] ribosome 2 present   par[24] ribosome moves are tried every so many sweeps
 * par[25] bond length between two paired beads (A)   par[26] bending stiffness of a paired stretch (kT)
 * par[27] step of a paired bead relative to that of a free bead
 *
 * stats[0] accepted bead moves / attempted   stats[1] accepted rigid moves / attempted
 * stats[2] sweep number of the first pairing (-1: none; kept across calls)   stats[3] paired beads at the end
 * stats[4] squared displacement of an unpaired bead per sweep (A^2)
 * stats[5] sweeps with at least one in-register interface contact while both chains are competent
 * stats[6] sweeps with the two N-terminal beads in contact   stats[7] interface contacts summed over pairs and sweeps
 * stats[8] sweep number of the first such contact (-1: none; kept across calls)
 * stats[9] squared displacement of a ribosome centre per sweep (A^2)   stats[10] squared rotation angle per sweep (rad^2)
 */
int run(const double *par,
        const float *g1, const double *o1, const int *n1, double h1,
        const float *g2, const double *o2, const int *n2, double h2,
        const double *pts2, int npts2,
        double *pose1, double *pose2, double *xa, int na, double *xb, int nb, int *bound,
        const double *exit1n, const double *exit2n, const double *q1n, const double *q2n,
        const double *centre1n, const double *centre2n,
        long nsweeps, double *stats, uint64_t *rng) {
    double b = par[0], K = par[1], step = par[2], cb = par[3], d0 = par[4], rc = par[5];
    int i0 = (int)par[6], i1 = (int)par[7], gate = (int)par[8];
    double pnuc = par[9], Lc = par[10], kt = par[11], dtr = par[12], drot = par[13], cclash = par[14], fclash = par[15];
    int attA = (int)par[16], attB = (int)par[17], tether = (int)par[18], sub = (int)par[19];
    int rib1 = (int)par[22], rib2 = (int)par[23], every = (int)par[24];
    if (every < 1) every = 1;
    double bh = par[25], kappa = par[26], fstep = par[27];
    World W = {{g1, o1, h1, n1}, {g2, o2, h2, n2}, pose1, pose2, rib1, rib2};
    if (sub < 1) sub = 1;
    /* the random stream continues across calls */
    if (rng[0] | rng[1] | rng[2] | rng[3]) memcpy(s, rng, sizeof(s)); else seed_rng((uint64_t)par[20]);
    long sweep0 = (long)par[21];
    long acc = 0, att = 0, racc = 0, ratt = 0, c_any = 0, c_nterm = 0, c_sum = 0, att_free = 0;
    double msd = 0.0, rmsd = 0.0, rang = 0.0, first = stats[2], first_contact = stats[8];
    int nbound = 0;
    for (int i = 0; i < i1; i++) nbound += bound[i];

    for (long sw = 0; sw < nsweeps; sw++) {
        double anchorA[3] = {0, 0, 0}, anchorB[3] = {0, 0, 0};
        if (rib1) to_lab(pose1, exit1n, anchorA);
        if (rib2) to_lab(pose2, exit2n, anchorB);
        /* ---- bead moves */
        for (int c = 0; c < 2; c++) {
            double *x = c ? xb : xa, *y = c ? xa : xb;
            int n = c ? nb : na, m = c ? na : nb, attached = c ? attB : attA;
            const double *anchor = c ? anchorB : anchorA;
            for (int i = 0; i < n; i++) {
                att++;
                int bi = bound[i];
                double st = bi ? step * fstep : step;
                if (!bi) att_free++;
                double *p = x + 3 * i, q[3] = {p[0] + sym(st), p[1] + sym(st), p[2] + sym(st)};
                if (!is_free(&W, q, cb)) continue;
                double dE = 0.0;
                if (i > 0) {
                    if (!mid_free(&W, q, x + 3 * (i - 1), cb)) continue;
                    double rest = (bi && bound[i - 1]) ? bh : b;
                    double r0 = dist(p, x + 3 * (i - 1)) - rest, r1 = dist(q, x + 3 * (i - 1)) - rest;
                    dE += 0.5 * K * (r1 * r1 - r0 * r0);
                }
                if (i < n - 1) {
                    if (!mid_free(&W, q, x + 3 * (i + 1), cb)) continue;
                    double rest = (bi && bound[i + 1]) ? bh : b;
                    double r0 = dist(p, x + 3 * (i + 1)) - rest, r1 = dist(q, x + 3 * (i + 1)) - rest;
                    dE += 0.5 * K * (r1 * r1 - r0 * r0);
                } else if (attached) {
                    double r0 = dist(p, anchor) - b, r1 = dist(q, anchor) - b;
                    dE += 0.5 * K * (r1 * r1 - r0 * r0);
                }
                if (bi && i < m) {
                    double r0 = dist(p, y + 3 * i) - d0, r1 = dist(q, y + 3 * i) - d0;
                    dE += 0.5 * K * (r1 * r1 - r0 * r0);
                }
                if (kappa > 0.0) {      /* bending of a paired stretch: the angles at beads i-1, i and i+1 */
                    for (int j = i - 1; j <= i + 1; j++) {
                        if (j < 1 || j + 1 >= n || !(bound[j - 1] && bound[j] && bound[j + 1])) continue;
                        const double *A = x + 3 * (j - 1), *B = x + 3 * j, *D = x + 3 * (j + 1);
                        dE += kappa * (cosang(A, B, D) - cosang(j - 1 == i ? q : A, j == i ? q : B, j + 1 == i ? q : D));
                    }
                }
                if (dE > 0.0 && urand() >= exp(-dE)) continue;
                if (!bi) msd += (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) + (q[2] - p[2]) * (q[2] - p[2]);
                p[0] = q[0]; p[1] = q[1]; p[2] = q[2]; acc++;
            }
        }
        /* ---- rigid-body moves of the ribosomes */
        for (int k = 0; k < 2; k++) {
            if ((sweep0 + sw) % every) break;
            if (!(k ? rib2 : rib1)) continue;
            double *pose = k ? pose2 : pose1, P[12], shift2, angle2;
            const double *Pa = k ? pose1 : P, *Pb = k ? P : pose2;      /* candidate poses of ribosome 1 and 2 */
            ratt++;
            if (!propose(pose, k ? centre2n : centre1n, dtr, drot, P, &shift2, &angle2)) continue;
            int ok = 1;
            double dE = 0.0;
            if (tether && rib1 && rib2) {
                double a0[3], b0[3], a1[3], b1[3];
                to_lab(pose1, q1n, a0); to_lab(pose2, q2n, b0); to_lab(Pa, q1n, a1); to_lab(Pb, q2n, b1);
                double r0 = dist(a0, b0), r1 = dist(a1, b1);
                if (r1 > Lc) ok = 0;
                dE += kt * (r1 * r1 - r0 * r0);
            }
            int attached = k ? attB : attA, n = k ? nb : na;
            const double *x = k ? xb : xa;
            if (ok && attached && n > 0) {
                double an[3]; to_lab(P, k ? exit2n : exit1n, an);
                const double *old = k ? anchorB : anchorA;
                double e0 = dist(x + 3 * (n - 1), old) - b, e1 = dist(x + 3 * (n - 1), an) - b;
                dE += 0.5 * K * (e1 * e1 - e0 * e0);
            }
            if (ok && dE > 0.0 && urand() >= exp(-dE)) ok = 0;
            if (ok) { /* chains may not end up inside the moved ribosome */
                const Grid *G = k ? &W.G2 : &W.G1;
                double w[3];
                for (int i = 0; i < na && ok; i++) { to_native(P, xa + 3 * i, w); if (clearance(G, w) < cb) ok = 0; }
                for (int i = 0; i < nb && ok; i++) { to_native(P, xb + 3 * i, w); if (clearance(G, w) < cb) ok = 0; }
            }
            if (ok && rib1 && rib2) { /* the two ribosomes may not overlap */
                int bad = 0;
                int limit = (int)(fclash * (npts2 / sub)) + 1;
                /* points of ribosome 2 in the frame of ribosome 1: v = M p + sh */
                double M[9], sh[3], v[3], dt3[3] = {Pb[9] - Pa[9], Pb[10] - Pa[10], Pb[11] - Pa[11]};
                for (int a = 0; a < 3; a++) {
                    for (int e = 0; e < 3; e++) M[3 * a + e] = Pa[a] * Pb[e] + Pa[3 + a] * Pb[3 + e] + Pa[6 + a] * Pb[6 + e];
                    sh[a] = Pa[a] * dt3[0] + Pa[3 + a] * dt3[1] + Pa[6 + a] * dt3[2];
                }
                for (int i = 0; i < npts2; i += sub) {
                    const double *q = pts2 + 3 * i;
                    for (int a = 0; a < 3; a++) v[a] = M[3 * a] * q[0] + M[3 * a + 1] * q[1] + M[3 * a + 2] * q[2] + sh[a];
                    if (clearance(&W.G1, v) < cclash && ++bad > limit) break;
                }
                if (bad > limit) ok = 0;
            }
            if (ok) { memcpy(pose, P, sizeof(P)); racc++; rmsd += shift2; rang += angle2; }
        }
        /* ---- contacts and pairing of the interface */
        int m = na < nb ? na : nb, hi = m < i1 ? m : i1;
        int expA = (na < i1 ? na : i1) - i0, expB = (nb < i1 ? nb : i1) - i0;
        int competent = expA >= gate && expB >= gate, any = 0;
        if (m > 0 && dist(xa, xb) < rc) c_nterm++;
        for (int i = i0; i < hi; i++) {
            if (bound[i]) continue;
            if (dist(xa + 3 * i, xb + 3 * i) >= rc) continue;
            if (competent) { c_sum++; any = 1; }
            int next_to = (i > 0 && bound[i - 1]) || (i + 1 < i1 && bound[i + 1]);
            if (nbound > 0 ? (next_to || competent) : (competent && pnuc > 0.0 && urand() < pnuc)) {
                bound[i] = 1; nbound++;
                if (first < 0) first = (double)(sweep0 + sw);
            }
        }
        if (any) { c_any++; if (first_contact < 0) first_contact = (double)(sweep0 + sw); }
    }
    int nrib = rib1 + rib2;
    stats[0] = att ? (double)acc / att : 0.0;
    stats[1] = ratt ? (double)racc / ratt : 0.0;
    stats[2] = first;
    stats[3] = nbound;
    stats[4] = att_free > 0 ? msd / (double)att_free : 0.0;
    stats[5] = (double)c_any;
    stats[6] = (double)c_nterm;
    stats[7] = (double)c_sum;
    stats[8] = first_contact;
    stats[9] = nrib > 0 && nsweeps > 0 ? rmsd / ((double)nrib * nsweeps) : 0.0;
    stats[10] = nrib > 0 && nsweeps > 0 ? rang / ((double)nrib * nsweeps) : 0.0;
    memcpy(rng, s, sizeof(s));
    return nbound;
}
