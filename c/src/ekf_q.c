#include "ekf_q.h"

#include <math.h>

#ifdef EKF_COUNT_OPS
ekf_ops_t ekf_ops;
#define CNT(f, n) (ekf_ops.f += (n))
#else
#define CNT(f, n) ((void)0)
#endif

#define P_LO (1 << 25)
#define P_HI (1 << 26)

/* ------------------------------------------------------------------ helpers */
int32_t sat32(int64_t x) {
    CNT(branch, 2);
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

int64_t shr_round(int64_t x, int sh) {
    CNT(shift, 1); CNT(add, 1);
    if (sh <= 0) return x;
    if (sh > 62) return 0;
    return (x + ((int64_t)1 << (sh - 1))) >> sh;
}

int64_t mul_shr(int64_t a, int64_t b, int sh) {
    CNT(mul, 1);
    return shr_round(a * b, sh);
}

static int msb64(uint64_t x) { /* index of highest set bit, -1 for 0 (CLZ on Cortex-M) */
    int n = -1;
    while (x) { x >>= 1; n++; }
    CNT(shift, 1);
    return n;
}

/* constant (m * 2^-e) expressed in P units at exponent pe: m * 2^(31 + pe - e) */
static int64_t const_in_p(q_const_t c, int32_t pe) {
    int sh = 31 + pe - c.e;
    CNT(shift, 1);
    if (sh >= 0) return (int64_t)c.m << (sh > 31 ? 31 : sh);
    return shr_round(c.m, -sh);
}

static void normalize(ekf_q_state_t *s, int64_t t[6]) {
    uint64_t mx = 0;
    for (int k = 0; k < 6; k++) {
        uint64_t a = (uint64_t)(t[k] < 0 ? -t[k] : t[k]);
        if (a > mx) mx = a;
        CNT(branch, 2);
    }
    if (mx == 0) { for (int k = 0; k < 6; k++) s->pq[k] = 0; return; }
    int sh = msb64(mx) - 25; /* want msb == 25 */
    for (int k = 0; k < 6; k++) s->pq[k] = sat32(sh > 0 ? shr_round(t[k], sh) : t[k] * ((int64_t)1 << -sh));
    /* the carries are fractions of one LSB (< 2^32): rescale them with P */
    if (sh > 0) { s->acc_q0 = shr_round(s->acc_q0, sh); s->acc_qr = shr_round(s->acc_qr, sh); }
    else if (sh < 0) { int l = -sh > 31 ? 31 : -sh; s->acc_q0 <<= l; s->acc_qr <<= l; }
    s->pe -= sh;
}

/* add constant c (in P units) with sub-LSB carry: returns whole LSBs to add, keeps remainder in *acc */
static int64_t add_with_carry(q_const_t c, int32_t pe, int64_t *acc) {
    int sh = 31 + pe - c.e + 32; /* c in units of 2^-32 LSB */
    CNT(shift, 2); CNT(add, 2);
    int64_t v;
    if (sh >= 0) {
        if (sh > 30 + 32 - 31) return const_in_p(c, pe); /* large constant: no carry needed */
        v = (int64_t)c.m << sh;
    } else {
        v = shr_round(c.m, -sh);
    }
    *acc += v;
    int64_t whole = *acc >> 32;
    *acc -= whole * ((int64_t)1 << 32);
    return whole;
}

#define SOC_MIN (-(1 << (SOC_FRAC - 1)))     /* -0.5 */
#define SOC_MAX ((3 << (SOC_FRAC - 1)))      /* +1.5 */
static int32_t clamp_soc(int32_t x) {
    CNT(branch, 2);
    return x < SOC_MIN ? SOC_MIN : (x > SOC_MAX ? SOC_MAX : x);
}

/* ------------------------------------------------------------------ table lookup */
typedef struct { int32_t ocv, docv, r0, dr0, b1, b2; } lk_t;

static lk_t lookup(const ekf_q_model_t *m, int32_t soc) {
    int64_t x = (int64_t)soc * (EKF_N - 1); /* Q.30 grid coordinate */
    CNT(mul, 1);
    int64_t k = x >> SOC_FRAC;
    if (k < 0) k = 0;
    if (k > EKF_N - 2) k = EKF_N - 2;
    int64_t f = x - (k << SOC_FRAC);            /* unclamped: OCV extrapolates */
    const int64_t FMAX = (int64_t)64 << SOC_FRAC; /* extrapolate at most 64 steps (32% SoC) */
    if (f > FMAX) f = FMAX;
    if (f < -FMAX) f = -FMAX;
    int64_t fc = f < 0 ? 0 : (f > ((int64_t)1 << SOC_FRAC) ? ((int64_t)1 << SOC_FRAC) : f);
    CNT(branch, 8); CNT(load, 12); CNT(add, 8);
    lk_t o;
    int i = (int)k;
    o.ocv = sat32(m->ocv[i] + mul_shr((int64_t)m->ocv[i + 1] - m->ocv[i], f, SOC_FRAC));
#define LIN(t) sat32(m->t[i] + mul_shr((int64_t)m->t[i + 1] - m->t[i], fc, SOC_FRAC))
    o.docv = LIN(docv); o.r0 = LIN(r0); o.dr0 = LIN(dr0); o.b1 = LIN(b1); o.b2 = LIN(b2);
#undef LIN
    return o;
}

/* ------------------------------------------------------------------ filter */
void ekf_q_init(ekf_q_state_t *s, const ekf_q_model_t *m, int32_t soc0) {
    s->soc = soc0; s->v1 = 0; s->v2 = 0; s->pe = 0; s->acc_q0 = 0; s->acc_qr = 0;
    int64_t t[6] = {const_in_p(m->p00_0, 0), 0, 0, const_in_p(m->prc_0, 0), 0, const_in_p(m->prc_0, 0)};
    normalize(s, t);
}

int32_t ekf_q_step(ekf_q_state_t *s, const ekf_q_model_t *m, int32_t i_avg, int32_t i_now, int32_t v, int predict) {
    int32_t *p = s->pq;
    int64_t t[6];
    if (predict) {
        lk_t L = lookup(m, s->soc);
        /* soc -= kq * I : (m*2^-e)(I*2^-25) -> 2^-30 units: shift right by e + 25 - 30 */
        s->soc = sat32((int64_t)s->soc - mul_shr(m->kq.m, i_avg, m->kq.e + I_FRAC - SOC_FRAC));
        s->v1 = sat32(mul_shr(m->a1, s->v1, 31) + mul_shr(L.b1, i_avg, R_FRAC + I_FRAC - V_FRAC));
        s->v2 = sat32(mul_shr(m->a2, s->v2, 31) + mul_shr(L.b2, i_avg, R_FRAC + I_FRAC - V_FRAC));
        CNT(add, 3);
        t[0] = p[0] + add_with_carry(m->q0, s->pe, &s->acc_q0);
        t[1] = mul_shr(p[1], m->a1, 31);
        t[2] = mul_shr(p[2], m->a2, 31);
        int64_t qr = add_with_carry(m->qr, s->pe, &s->acc_qr);
        t[3] = mul_shr(p[3], m->a1sq, 31) + qr;
        t[4] = mul_shr(p[4], m->a1a2, 31);
        t[5] = mul_shr(p[5], m->a2sq, 31) + qr;
        CNT(add, 3);
        normalize(s, t);
    }
    lk_t L = lookup(m, s->soc);
    /* h0 = dOCV - dR0 * I  (Q4.27) ; prediction error e (Q3.28) */
    int64_t h0 = sat32((int64_t)L.docv - mul_shr(L.dr0, i_now, I_FRAC));
    int64_t ypred = (int64_t)L.ocv - mul_shr(L.r0, i_now, R_FRAC + I_FRAC - V_FRAC) - s->v1 - s->v2;
    int64_t e = sat32((int64_t)v - ypred);
    /* P H^T and innovation variance, in P units (|pq| < 2^26, |h0| < 2^31 => products < 2^30) */
    int64_t ph0 = mul_shr(p[0], h0, S_FRAC) - p[1] - p[2];
    int64_t ph1 = mul_shr(p[1], h0, S_FRAC) - p[3] - p[4];
    int64_t ph2 = mul_shr(p[2], h0, S_FRAC) - p[4] - p[5];
    int64_t sv = mul_shr(h0, ph0, S_FRAC) - ph1 - ph2 + const_in_p(m->r, s->pe);
    CNT(add, 12);
    CNT(branch, 1);
    if (sv <= 0) return s->soc; /* numerically lost: skip the update */
    /* K = PH^T / S in Q5.26 */
    int32_t k0 = sat32((ph0 * ((int64_t)1 << K_FRAC)) / sv);
    int32_t k1 = sat32((ph1 * ((int64_t)1 << K_FRAC)) / sv);
    int32_t k2 = sat32((ph2 * ((int64_t)1 << K_FRAC)) / sv);
    CNT(div, 3); CNT(shift, 3);
    /* x += K e   (Q5.26 * Q3.28 -> Q1.30: >> 24 ; -> Q3.28: >> 26) */
    s->soc = clamp_soc(sat32((int64_t)s->soc + mul_shr(k0, e, K_FRAC + V_FRAC - SOC_FRAC)));
    s->v1 = sat32((int64_t)s->v1 + mul_shr(k1, e, K_FRAC));
    s->v2 = sat32((int64_t)s->v2 + mul_shr(k2, e, K_FRAC));
    /* P -= K (PH^T)^T */
    t[0] = p[0] - mul_shr(k0, ph0, K_FRAC);
    t[1] = p[1] - mul_shr(k0, ph1, K_FRAC);
    t[2] = p[2] - mul_shr(k0, ph2, K_FRAC);
    t[3] = p[3] - mul_shr(k1, ph1, K_FRAC);
    t[4] = p[4] - mul_shr(k1, ph2, K_FRAC);
    t[5] = p[5] - mul_shr(k2, ph2, K_FRAC);
    CNT(add, 9); CNT(branch, 3);
    if (t[0] < 1) t[0] = 1; /* keep variances positive */
    if (t[3] < 1) t[3] = 1;
    if (t[5] < 1) t[5] = 1;
    normalize(s, t);
    return s->soc;
}

double ekf_q_p00(const ekf_q_state_t *s) { return ldexp((double)s->pq[0], -(31 + s->pe)); }

/* ------------------------------------------------------------------ quantisation (start-up only) */
int32_t q_from_double(double x, int frac) {
    double y = ldexp(x, frac);
    y = y >= 0 ? floor(y + 0.5) : ceil(y - 0.5);
    if (y > INT32_MAX) return INT32_MAX;
    if (y < INT32_MIN) return INT32_MIN;
    return (int32_t)y;
}

double q_to_double(int32_t x, int frac) { return ldexp((double)x, -frac); }

static q_const_t qc(double v) {
    q_const_t c = {0, 0};
    if (v <= 0) return c;
    int ex;
    frexp(v, &ex);          /* v = f * 2^ex, f in [0.5, 1) */
    c.e = 31 - ex;          /* m = v * 2^e in [2^30, 2^31) */
    double m = floor(ldexp(v, c.e) + 0.5);
    if (m >= 2147483648.0) { m /= 2; c.e -= 1; }
    c.m = (int32_t)m;
    return c;
}

void ekf_q_from_f(const ekf_f_model_t *f, ekf_q_model_t *q) {
    double a1 = exp(-f->dt / f->tau1), a2 = exp(-f->dt / f->tau2);
    for (int k = 0; k < EKF_N; k++) {
        q->ocv[k] = q_from_double(f->ocv[k], V_FRAC);
        q->docv[k] = q_from_double(f->docv[k], S_FRAC);
        q->r0[k] = q_from_double(f->r0[k], R_FRAC);
        q->dr0[k] = q_from_double(f->dr0[k], S_FRAC);
        q->b1[k] = q_from_double(f->r1[k] * (1 - a1), R_FRAC);
        q->b2[k] = q_from_double(f->r2[k] * (1 - a2), R_FRAC);
    }
    q->a1 = q_from_double(a1, 31); q->a2 = q_from_double(a2, 31);
    q->a1sq = q_from_double(a1 * a1, 31); q->a2sq = q_from_double(a2 * a2, 31);
    q->a1a2 = q_from_double(a1 * a2, 31);
    q->kq = qc(f->dt / (3600.0 * f->capacity_ah));
    q->q0 = qc(f->sigma_soc_proc * f->sigma_soc_proc * f->dt);
    q->qr = qc(f->sigma_rc_proc * f->sigma_rc_proc * f->dt);
    q->r = qc(f->sigma_v * f->sigma_v);
    q->p00_0 = qc(f->sigma_soc0 * f->sigma_soc0);
    q->prc_0 = qc(f->sigma_rc0 * f->sigma_rc0);
}
