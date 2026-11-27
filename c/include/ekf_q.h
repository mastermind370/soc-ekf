/* Fixed-point EKF for SoC (2-RC model). int32 storage, int64 intermediates, saturating arithmetic.
 *
 * Formats (Qm.n = m integer bits + n fraction bits in an int32, sign included in m):
 *   SoC                 Q1.30   (+-2,   LSB 9.3e-10)
 *   voltages, OCV, e    Q3.28   (+-8 V, LSB 3.7 nV)
 *   current             Q6.25   (+-64 A, LSB 30 nA)
 *   R0, b_k=R_k(1-a_k)  Q0.31   (0..1 ohm)
 *   a1, a2, a^2, a1a2   Q0.31
 *   slopes dOCV, dR0, h Q4.27   (+-16 per unit SoC)
 *   Kalman gain K       Q5.26   (+-32)
 *   covariance P        block floating point: P = pq * 2^-(31+pe), |pq| kept in [2^25, 2^26)
 *   positive constants  mantissa/exponent: value = m * 2^-e, m in [2^30, 2^31)
 * Process noise q0, qr can be far below one P LSB (q0 = 1e-12 vs P ~ 1e-4); a carry accumulator keeps the
 * remainder so it is added exactly over time instead of being rounded away.
 * Fault guard: SoC clamped to [-0.5, 1.5] (never reached in normal operation).
 * No floating point inside ekf_q_step(). Division: 3 x int64/int64 per update.
 */
#ifndef EKF_Q_H
#define EKF_Q_H

#include <stdint.h>

#include "ekf_f.h"

#define SOC_FRAC 30
#define V_FRAC 28
#define I_FRAC 25
#define R_FRAC 31
#define S_FRAC 27
#define K_FRAC 26

typedef struct { int32_t m; int32_t e; } q_const_t;

typedef struct {
    int32_t ocv[EKF_N], docv[EKF_N], r0[EKF_N], dr0[EKF_N], b1[EKF_N], b2[EKF_N];
    int32_t a1, a2, a1sq, a2sq, a1a2; /* Q0.31 */
    q_const_t kq;                     /* dt / (3600 Q)          [SoC per A] */
    q_const_t q0, qr, r;              /* process / measurement noise variances */
    q_const_t p00_0, prc_0;           /* initial variances */
} ekf_q_model_t;

typedef struct {
    int32_t soc, v1, v2;
    int32_t pq[6]; /* p00 p01 p02 p11 p12 p22 */
    int32_t pe;
    int64_t acc_q0, acc_qr; /* sub-LSB remainders of the process noise, in units of 2^-32 LSB */
} ekf_q_state_t;

/* Offline / start-up: quantise a double model (not used inside the loop). */
void ekf_q_from_f(const ekf_f_model_t *f, ekf_q_model_t *q);
int32_t q_from_double(double x, int frac); /* round + saturate */
double q_to_double(int32_t x, int frac);

void ekf_q_init(ekf_q_state_t *s, const ekf_q_model_t *m, int32_t soc0);
int32_t ekf_q_step(ekf_q_state_t *s, const ekf_q_model_t *m, int32_t i_avg, int32_t i_now, int32_t v, int predict);
double ekf_q_p00(const ekf_q_state_t *s); /* covariance read-out for tests */

/* saturating helpers (exposed for unit tests) */
int32_t sat32(int64_t x);
int64_t mul_shr(int64_t a, int64_t b, int sh); /* round(a*b / 2^sh); caller keeps |a*b| < 2^63 */
int64_t shr_round(int64_t x, int sh);

#ifdef EKF_COUNT_OPS
typedef struct { long mul, add, shift, div, load, branch; } ekf_ops_t;
extern ekf_ops_t ekf_ops;
#endif

#endif
