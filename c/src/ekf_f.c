#include "ekf_f.h"

#include <math.h>

typedef struct { double ocv, docv, r0, dr0, r1, r2; } lk_t;

static lk_t lookup(const ekf_f_model_t *m, double s) {
    const double step = 1.0 / (EKF_N - 1);
    double x = s / step;
    int k = x >= 0 ? (int)x : 0;
    if (k > EKF_N - 2) k = EKF_N - 2;
    double f = x - k;
    double sc = f < 0 ? 0.0 : (f > 1 ? 1.0 : f);
    lk_t o;
    o.ocv = m->ocv[k] + (m->ocv[k + 1] - m->ocv[k]) * f;
#define LIN(t) (m->t[k] + (m->t[k + 1] - m->t[k]) * sc)
    o.docv = LIN(docv); o.r0 = LIN(r0); o.dr0 = LIN(dr0); o.r1 = LIN(r1); o.r2 = LIN(r2);
#undef LIN
    return o;
}

void ekf_f_init(ekf_f_state_t *s, const ekf_f_model_t *m, double soc0) {
    s->soc = soc0; s->v1 = 0; s->v2 = 0;
    s->p00 = m->sigma_soc0 * m->sigma_soc0; s->p01 = 0; s->p02 = 0;
    s->p11 = m->sigma_rc0 * m->sigma_rc0; s->p12 = 0; s->p22 = s->p11;
}

double ekf_f_step(ekf_f_state_t *s, const ekf_f_model_t *m, double i_avg, double i_now, double v, int predict) {
    const double dt = m->dt;
    const double a1 = exp(-dt / m->tau1), a2 = exp(-dt / m->tau2);
    if (predict) {
        lk_t L = lookup(m, s->soc);
        double q0 = m->sigma_soc_proc * m->sigma_soc_proc * dt, qr = m->sigma_rc_proc * m->sigma_rc_proc * dt;
        s->soc -= dt / (3600.0 * m->capacity_ah) * i_avg;
        s->v1 = a1 * s->v1 + L.r1 * (1.0 - a1) * i_avg;
        s->v2 = a2 * s->v2 + L.r2 * (1.0 - a2) * i_avg;
        s->p00 += q0; s->p01 *= a1; s->p02 *= a2;
        s->p11 = a1 * a1 * s->p11 + qr; s->p12 = a1 * a2 * s->p12; s->p22 = a2 * a2 * s->p22 + qr;
    }
    lk_t L = lookup(m, s->soc);
    double h0 = L.docv - L.dr0 * i_now;
    double e = v - (L.ocv - L.r0 * i_now - s->v1 - s->v2);
    double ph0 = s->p00 * h0 - s->p01 - s->p02;
    double ph1 = s->p01 * h0 - s->p11 - s->p12;
    double ph2 = s->p02 * h0 - s->p12 - s->p22;
    double sv = h0 * ph0 - ph1 - ph2 + m->sigma_v * m->sigma_v;
    double k0 = ph0 / sv, k1 = ph1 / sv, k2 = ph2 / sv;
    s->soc += k0 * e; s->v1 += k1 * e; s->v2 += k2 * e;
    s->p00 -= k0 * ph0; s->p01 -= k0 * ph1; s->p02 -= k0 * ph2;
    s->p11 -= k1 * ph1; s->p12 -= k1 * ph2; s->p22 -= k2 * ph2;
    return s->soc;
}
