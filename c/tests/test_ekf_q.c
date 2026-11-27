/* Unit + stress tests for the fixed-point EKF. Build with -fsanitize=undefined to catch any signed overflow. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ekf_f.h"
#include "ekf_q.h"

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void make_model(ekf_f_model_t *m) {
    for (int k = 0; k < EKF_N; k++) {
        double s = (double)k / (EKF_N - 1);
        m->ocv[k] = 3.0 + 1.1 * s + 0.15 * tanh(20 * (s - 0.05));
        m->r0[k] = 0.03 - 0.01 * s;
        m->r1[k] = 0.012; m->r2[k] = 0.015;
    }
    for (int k = 0; k < EKF_N; k++) { /* numpy.gradient equivalent */
        int a = k > 0 ? k - 1 : 0, b = k < EKF_N - 1 ? k + 1 : EKF_N - 1;
        double ds = (double)(b - a) / (EKF_N - 1);
        m->docv[k] = (m->ocv[b] - m->ocv[a]) / ds;
        m->dr0[k] = (m->r0[b] - m->r0[a]) / ds;
    }
    m->tau1 = 9; m->tau2 = 300; m->capacity_ah = 2.9; m->dt = 1.0;
    m->sigma_v = 0.01; m->sigma_soc_proc = 1e-5; m->sigma_rc_proc = 1e-3; m->sigma_soc0 = 0.25; m->sigma_rc0 = 0.01;
}

static void test_helpers(void) {
    CHECK(sat32((int64_t)INT32_MAX + 1) == INT32_MAX, "sat high");
    CHECK(sat32((int64_t)INT32_MIN - 5) == INT32_MIN, "sat low");
    CHECK(sat32(123) == 123, "sat pass");
    CHECK(shr_round(5, 1) == 3 && shr_round(-5, 1) == -2 && shr_round(4, 1) == 2, "round half up");
    CHECK(shr_round(12345, 0) == 12345 && shr_round(1, 63) == 0, "shift edge");
    CHECK(mul_shr(1 << 30, 1 << 30, 30) == (1 << 30), "mul 1*1 in Q30");
    CHECK(q_from_double(1e9, 28) == INT32_MAX && q_from_double(-1e9, 28) == INT32_MIN, "q_from_double saturates");
    CHECK(fabs(q_to_double(q_from_double(3.7, V_FRAC), V_FRAC) - 3.7) < 1e-8, "V round trip");
    CHECK(fabs(q_to_double(q_from_double(-17.25, I_FRAC), I_FRAC) + 17.25) < 1e-7, "I round trip");
}

static void run_profile(const ekf_f_model_t *mf, const ekf_q_model_t *mq, double soc0_est, int n,
                        double *max_diff, double *final_err) {
    /* synthetic cell = the model itself, simulated in double */
    double soc = 1.0, v1 = 0, v2 = 0, a1 = exp(-1.0 / mf->tau1), a2 = exp(-1.0 / mf->tau2);
    ekf_f_state_t sf; ekf_q_state_t sq;
    ekf_f_init(&sf, mf, soc0_est);
    ekf_q_init(&sq, mq, q_from_double(soc0_est, SOC_FRAC));
    srand(7);
    *max_diff = 0;
    double ef = 0;
    for (int k = 0; k < n; k++) {
        double I = 1.0 + 2.5 * sin(k / 31.0) * sin(k / 9.7) + 0.3 * ((double)rand() / RAND_MAX - 0.5);
        if (k > 0) {
            int g = (int)(soc * 200); if (g < 0) g = 0; if (g > 199) g = 199;
            soc -= I / (3600.0 * mf->capacity_ah);
            v1 = a1 * v1 + mf->r1[g] * (1 - a1) * I;
            v2 = a2 * v2 + mf->r2[g] * (1 - a2) * I;
        }
        int g = (int)(soc * 200); if (g < 0) g = 0; if (g > 199) g = 199;
        double V = mf->ocv[g] + (mf->ocv[g + 1] - mf->ocv[g]) * (soc * 200 - g) - mf->r0[g] * I - v1 - v2;
        double f = ekf_f_step(&sf, mf, I, I, V, k > 0);
        int32_t qv = ekf_q_step(&sq, mq, q_from_double(I, I_FRAC), q_from_double(I, I_FRAC), q_from_double(V, V_FRAC), k > 0);
        double d = fabs(q_to_double(qv, SOC_FRAC) - f);
        if (d > *max_diff) *max_diff = d;
        ef = f - soc;
    }
    *final_err = ef;
}

static void test_tracks_float(void) {
    ekf_f_model_t mf; ekf_q_model_t mq;
    make_model(&mf); ekf_q_from_f(&mf, &mq);
    double md, fe;
    run_profile(&mf, &mq, 1.0, 9000, &md, &fe);
    printf("  correct start: max |Q - float| = %.4f %% SoC, float final error %.3f %%\n", 100 * md, 100 * fe);
    CHECK(md < 0.002, "Q tracks float from true start (max diff %.5f)", md);
    CHECK(fabs(fe) < 0.01, "float EKF converged (%.4f)", fe);
    run_profile(&mf, &mq, 0.7, 9000, &md, &fe);
    printf("  30%% wrong start: max |Q - float| = %.4f %% SoC, float final error %.3f %%\n", 100 * md, 100 * fe);
    CHECK(md < 0.01, "Q tracks float from wrong start (max diff %.5f)", md);
    /* the noise levels the stage-4 grid search actually picks (smallest SoC process noise) */
    mf.sigma_soc_proc = 1e-6; mf.sigma_v = 0.04; ekf_q_from_f(&mf, &mq);
    run_profile(&mf, &mq, 0.7, 9000, &md, &fe); /* 9000 s: stays above 0% SoC */
    printf("  sigma_soc_proc=1e-6, sigma_v=0.04, wrong start: max |Q - float| = %.4f %% SoC\n", 100 * md);
    CHECK(md < 1e-4, "Q tracks float with tiny process noise (max diff %.6f)", md);
}

static void test_tiny_noise_not_lost(void) {
    /* sigma_soc_proc = 1e-6 -> q0 = 1e-12: plain Q31 would round it to 0; block-float P must keep it */
    ekf_f_model_t mf; ekf_q_model_t mq;
    make_model(&mf); mf.sigma_soc_proc = 1e-6; mf.sigma_soc0 = 1e-5; mf.sigma_v = 1.0; /* p00 << p11, p22 */
    ekf_q_from_f(&mf, &mq);
    ekf_q_state_t s; ekf_f_state_t f;
    ekf_q_init(&s, &mq, q_from_double(0.5, SOC_FRAC)); ekf_f_init(&f, &mf, 0.5);
    for (int k = 0; k < 2000; k++) {  /* predict-only growth: huge measurement noise via flat update */
        ekf_q_step(&s, &mq, 0, 0, q_from_double(mf.ocv[100], V_FRAC), 1);
        ekf_f_step(&f, &mf, 0, 0, mf.ocv[100], 1);
    }
    double rel = fabs(ekf_q_p00(&s) - f.p00) / f.p00;
    printf("  tiny-variance P00: float %.3e  fixed %.3e  (rel err %.2e)\n", f.p00, ekf_q_p00(&s), rel);
    /* P00 is ~2e-5 of the largest P entry here, so it only has ~10 mantissa bits (shared exponent):
       a documented dynamic-range limit. Without the carry this was 13%; with it a few %. */
    CHECK(rel < 0.10, "P00 with q0=1e-12 tracked (rel %.3e)", rel);
}

static void test_saturation_stress(void) {
    /* garbage / extreme sensor values for many steps: no UB (UBSan build), bounded state, P diag >= 0,
       and the filter works normally after re-initialisation */
    ekf_f_model_t mf; ekf_q_model_t mq;
    make_model(&mf); ekf_q_from_f(&mf, &mq);
    ekf_q_state_t s;
    ekf_q_init(&s, &mq, q_from_double(0.5, SOC_FRAC));
    const int32_t I[] = {INT32_MAX, INT32_MIN, 0, q_from_double(60, I_FRAC), q_from_double(-60, I_FRAC)};
    const int32_t V[] = {INT32_MAX, INT32_MIN, 0, q_from_double(4.2, V_FRAC), q_from_double(2.5, V_FRAC)};
    int ok_p = 1, ok_soc = 1;
    for (int k = 0; k < 100000; k++) {
        int32_t soc = ekf_q_step(&s, &mq, I[k % 5], I[(k / 5) % 5], V[(k / 3) % 5], 1);
        if (s.pq[0] < 0 || s.pq[3] < 0 || s.pq[5] < 0) ok_p = 0;
        if (soc < -(1 << 29) || soc > (3 << 29)) ok_soc = 0;
    }
    CHECK(ok_p, "variances stay non-negative under extreme inputs");
    CHECK(ok_soc, "SoC stays inside the [-0.5, 1.5] fault clamp");
    ekf_q_init(&s, &mq, q_from_double(0.3, SOC_FRAC));
    for (int k = 0; k < 600; k++)
        ekf_q_step(&s, &mq, 0, 0, q_from_double(mf.ocv[120], V_FRAC), k > 0);
    double soc = q_to_double(s.soc, SOC_FRAC);
    printf("  after stress, re-init at 30%% + 600 s at rest on OCV(60%%): SoC = %.4f\n", soc);
    CHECK(fabs(soc - 0.60) < 0.005, "re-initialised filter converges (%.4f)", soc);
}

static void test_lookup_extrapolation(void) {
    ekf_f_model_t mf; ekf_q_model_t mq;
    make_model(&mf); ekf_q_from_f(&mf, &mq);
    ekf_q_state_t s; ekf_f_state_t f;
    for (double s0 = -0.2; s0 <= 1.21; s0 += 0.1) { /* single update from many starting points */
        ekf_q_init(&s, &mq, q_from_double(s0, SOC_FRAC)); ekf_f_init(&f, &mf, s0);
        double a = q_to_double(ekf_q_step(&s, &mq, 0, 0, q_from_double(3.7, V_FRAC), 0), SOC_FRAC);
        double b = ekf_f_step(&f, &mf, 0, 0, 3.7, 0);
        CHECK(fabs(a - b) < 1e-4, "first update from soc0=%.2f: q %.6f f %.6f", s0, a, b);
    }
}

int main(void) {
    printf("fixed-point EKF tests\n");
    test_helpers();
    test_lookup_extrapolation();
    test_tracks_float();
    test_tiny_noise_not_lost();
    test_saturation_stress();
    printf("%s: %d/%d checks passed\n", fails ? "FAILED" : "ALL PASSED", checks - fails, checks);
    return fails ? 1 : 0;
}
