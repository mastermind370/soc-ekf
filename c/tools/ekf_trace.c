#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _POSIX_C_SOURCE 199309L
#endif
/* Run the fixed-point and double EKFs on a trace exported by scripts/export_c.py, and time them.
 *
 * usage: ekf_trace <model.bin> <trace.bin> <out.bin>
 *   model.bin : ekf_f_model_t written field by field as float64 (see export_c.py)
 *   trace.bin : int64 n, float64 soc0_true, float64 soc0_wrong, then n rows of (i_avg, i_now, v)
 *   out.bin   : n rows of float64 (q_true, q_wrong, f_true, f_wrong)
 * prints: ns_per_update_q ns_per_update_f
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ekf_f.h"
#include "ekf_q.h"

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int read_model(const char *path, ekf_f_model_t *m) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = 0;
    double *tabs[] = {m->ocv, m->docv, m->r0, m->dr0, m->r1, m->r2};
    for (int t = 0; t < 6; t++) n += fread(tabs[t], sizeof(double), EKF_N, f);
    double *sc[] = {&m->tau1, &m->tau2, &m->capacity_ah, &m->dt, &m->sigma_v, &m->sigma_soc_proc,
                    &m->sigma_rc_proc, &m->sigma_soc0, &m->sigma_rc0};
    for (int k = 0; k < 9; k++) n += fread(sc[k], sizeof(double), 1, f);
    fclose(f);
    return n == 6 * EKF_N + 9;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model.bin trace.bin out.bin\n", argv[0]); return 2; }
    ekf_f_model_t mf; ekf_q_model_t mq;
    if (!read_model(argv[1], &mf)) { fprintf(stderr, "bad model file\n"); return 2; }
    ekf_q_from_f(&mf, &mq);
    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "no trace\n"); return 2; }
    int64_t n; double s0[2];
    if (fread(&n, 8, 1, f) != 1 || fread(s0, 8, 2, f) != 2) return 2;
    double *in = malloc(sizeof(double) * 3 * n);
    if (fread(in, sizeof(double), 3 * n, f) != (size_t)(3 * n)) return 2;
    fclose(f);
    /* sensor boundary: convert once to fixed point (an ADC would deliver integers directly) */
    int32_t *qi = malloc(sizeof(int32_t) * 3 * n);
    for (int64_t k = 0; k < n; k++) {
        qi[3 * k] = q_from_double(in[3 * k], I_FRAC);
        qi[3 * k + 1] = q_from_double(in[3 * k + 1], I_FRAC);
        qi[3 * k + 2] = q_from_double(in[3 * k + 2], V_FRAC);
    }
    double *out = malloc(sizeof(double) * 4 * n);
    for (int w = 0; w < 2; w++) {
        ekf_q_state_t sq; ekf_f_state_t sf;
        ekf_q_init(&sq, &mq, q_from_double(s0[w], SOC_FRAC));
        ekf_f_init(&sf, &mf, s0[w]);
        for (int64_t k = 0; k < n; k++) {
            out[4 * k + w] = q_to_double(ekf_q_step(&sq, &mq, qi[3 * k], qi[3 * k + 1], qi[3 * k + 2], k > 0), SOC_FRAC);
            out[4 * k + 2 + w] = ekf_f_step(&sf, &mf, in[3 * k], in[3 * k + 1], in[3 * k + 2], k > 0);
        }
    }
    /* timing: repeat the true-start run until >= 2e6 updates */
    int reps = (int)(2000000 / n) + 1;
    volatile int32_t sink = 0;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        ekf_q_state_t sq;
        ekf_q_init(&sq, &mq, q_from_double(s0[0], SOC_FRAC));
        for (int64_t k = 0; k < n; k++) sink += ekf_q_step(&sq, &mq, qi[3 * k], qi[3 * k + 1], qi[3 * k + 2], k > 0);
    }
    double tq = (now_ns() - t0) / ((double)reps * n);
    volatile double fsink = 0;
    t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        ekf_f_state_t sf;
        ekf_f_init(&sf, &mf, s0[0]);
        for (int64_t k = 0; k < n; k++) fsink += ekf_f_step(&sf, &mf, in[3 * k], in[3 * k + 1], in[3 * k + 2], k > 0);
    }
    double tf = (now_ns() - t0) / ((double)reps * n);
#ifdef EKF_COUNT_OPS
    ekf_ops = (ekf_ops_t){0};
    {
        ekf_q_state_t sq;
        ekf_q_init(&sq, &mq, q_from_double(s0[0], SOC_FRAC));
        ekf_ops = (ekf_ops_t){0};
        for (int64_t k = 0; k < n; k++) ekf_q_step(&sq, &mq, qi[3 * k], qi[3 * k + 1], qi[3 * k + 2], k > 0);
        printf("ops_per_update mul=%.1f add=%.1f shift=%.1f div=%.1f load=%.1f branch=%.1f\n",
               (double)ekf_ops.mul / n, (double)ekf_ops.add / n, (double)ekf_ops.shift / n,
               (double)ekf_ops.div / n, (double)ekf_ops.load / n, (double)ekf_ops.branch / n);
    }
#endif
    FILE *o = fopen(argv[3], "wb");
    fwrite(out, sizeof(double), 4 * n, o);
    fclose(o);
    printf("ns_per_update_q %.2f ns_per_update_f %.2f\n", tq, tf);
    free(in); free(qi); free(out);
    return 0;
}
