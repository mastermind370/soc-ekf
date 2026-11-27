/* Double-precision reference EKF: line-for-line the Python soc_ekf/ekf.py, used to check the C build. */
#ifndef EKF_F_H
#define EKF_F_H

#define EKF_N 201 /* uniform SoC grid 0..1, step 1/200 */

typedef struct {
    double ocv[EKF_N], docv[EKF_N], r0[EKF_N], dr0[EKF_N], r1[EKF_N], r2[EKF_N];
    double tau1, tau2, capacity_ah, dt;
    double sigma_v, sigma_soc_proc, sigma_rc_proc, sigma_soc0, sigma_rc0;
} ekf_f_model_t;

typedef struct {
    double soc, v1, v2;
    double p00, p01, p02, p11, p12, p22;
} ekf_f_state_t;

void ekf_f_init(ekf_f_state_t *s, const ekf_f_model_t *m, double soc0);
/* predict (unless first step) then update; returns the SoC estimate */
double ekf_f_step(ekf_f_state_t *s, const ekf_f_model_t *m, double i_avg, double i_now, double v, int predict);

#endif
