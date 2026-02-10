# FILE: ./src/gamma_transition.jl
"""
GAMMA_TRANSITION.JL - γ Intermittency Transition Model for LBM

Implements a simplified γ (gamma) transition model based on the Langtry-Menter
framework, adapted for LBM-LES. The intermittency factor γ ∈ [0,1] controls
the laminar-to-turbulent transition:
  - γ = 0: Fully laminar (no turbulent viscosity)
  - γ = 1: Fully turbulent (full WALE eddy viscosity)

The model solves a transport equation for γ using finite-difference
advection-diffusion with production/destruction source terms:
  ∂γ/∂t + u·∇γ = ∇·(D_γ ∇γ) + P_γ - E_γ

Production is triggered when the local vorticity Reynolds number exceeds
a critical threshold, driving γ toward 1. Destruction (relaminarization)
is active when turbulent activity is low.

Reference:
  Langtry & Menter, "Correlation-Based Transition Modeling for
  Unstructured Parallelized CFD Codes", AIAA J. 47(12), 2009.
"""

using KernelAbstractions

# ==============================================================================
# MODEL CONSTANTS
# ==============================================================================

# Production constants
const GAMMA_CA1 = 2.0f0        # Production multiplier
const GAMMA_CE1 = 1.0f0        # Production limiter
const GAMMA_CA2 = 0.06f0       # Destruction multiplier
const GAMMA_CE2 = 50.0f0       # Destruction limiter

# Onset function exponents
const GAMMA_ONSET_EXP = 3.0f0  # Sharpness of onset switch

# Diffusion coefficient for γ transport (lattice units)
const GAMMA_SIGMA = 1.0f0      # Diffusion coefficient σ_γ

# ==============================================================================
# TRANSITION ONSET CORRELATIONS
# ==============================================================================

"""
Compute the critical momentum-thickness Reynolds number Re_θc
from the freestream turbulence intensity Tu [%].

Menter correlation (simplified):
  Tu ≤ 1.3%: Re_θc = 803.73 * (Tu + 0.6067)^(-1.027)
  Tu > 1.3%: Re_θc = 331.50 * (Tu - 0.5658)^(-0.671)
"""
@inline function compute_re_theta_c(tu_percent::Float32)
    if tu_percent <= 1.3f0
        return 803.73f0 * (tu_percent + 0.6067f0)^(-1.027f0)
    else
        return 331.50f0 * (max(tu_percent - 0.5658f0, 0.01f0))^(-0.671f0)
    end
end

"""
Compute the transition length function F_length from Re_θc.

Controls how rapidly γ grows after onset. Higher F_length = faster transition.
Simplified piecewise correlation.
"""
@inline function compute_f_length(re_theta_c::Float32)
    if re_theta_c < 400.0f0
        return 398.189f0 - 119.270f0 * re_theta_c * 0.01f0 + 132.567f0 * (re_theta_c * 0.01f0)^2
    elseif re_theta_c < 596.0f0
        return 263.404f0 - 123.939f0 * re_theta_c * 0.01f0 + 194.584f0 * (re_theta_c * 0.01f0)^2 - 101.695f0 * (re_theta_c * 0.01f0)^3
    elseif re_theta_c < 1200.0f0
        return 0.5f0 - 3.0f-4 * (re_theta_c - 596.0f0)
    else
        return 0.3188f0
    end
end

"""
Compute the onset function F_onset that triggers transition.

Based on the ratio of local vorticity Reynolds number to critical value:
  R_v = ρ * |S| * d² / μ   (vorticity Reynolds number)
  F_onset1 = R_v / (2.193 * Re_θc)
  F_onset = max(F_onset1, F_onset1⁴, F_onset1_clamped)

Returns a value that rapidly switches from 0 (laminar) to large (transitional).
"""
@inline function compute_f_onset(rev::Float32, re_theta_c::Float32, r_t::Float32)
    f_onset1 = rev / (2.193f0 * max(re_theta_c, 1.0f0))
    f_onset2 = min(max(f_onset1, f_onset1 * f_onset1 * f_onset1 * f_onset1), 2.0f0)
    f_onset3 = max(1.0f0 - (r_t / 2.5f0)^3, 0.0f0)
    return max(f_onset2 - f_onset3, 0.0f0)
end

"""
Compute the turbulence suppression function F_turb.

Active in fully turbulent regions, drives destruction of γ if it overshoots.
  F_turb = exp(-(R_T / 4)⁴)
"""
@inline function compute_f_turb(r_t::Float32)
    ratio = r_t * 0.25f0
    r4 = ratio * ratio * ratio * ratio
    return exp(-min(r4, 20.0f0))
end

# ==============================================================================
# GAMMA NEIGHBOR ACCESS (reusing block topology)
# ==============================================================================

@inline function get_gamma_neighbor(gamma_arr, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)

    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return gamma_arr[nx, ny, nz, ib]
    end

    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]

    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return gamma_arr[nnx, nny, nnz, nb_idx]
    end

    # At domain boundary: Neumann (zero gradient) → return own value
    return gamma_arr[x, y, z, ib]
end

# ==============================================================================
# GAMMA TRANSPORT KERNEL
# ==============================================================================

"""
Kernel to update the intermittency field γ using finite-difference
advection-diffusion with Langtry-Menter source terms.

This kernel runs AFTER the main LBM stream-collide, using the updated
velocity and density fields.
"""
@kernel function gamma_transport_kernel!(
    gamma_out, gamma_in,
    vel, rho_arr, obstacle,
    wall_dist_arr, neighbor_table,
    active_coords_x, active_coords_y, active_coords_z,
    tau_molecular::Float32,
    re_theta_c::Float32,
    f_length::Float32,
    tu_intensity::Float32,
    gamma_diffusion::Float32,
    n_blocks::Int32,
    block_size::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)

    if b_idx <= n_blocks
        @inbounds begin
            is_obs = obstacle[x, y, z, b_idx]

            if is_obs
                # Solid cells: γ = 0 (no turbulence inside obstacle)
                gamma_out[x, y, z, b_idx] = 0.0f0
            else
                gamma_local = gamma_in[x, y, z, b_idx]

                # --- Local flow quantities ---
                ux = vel[x, y, z, b_idx, 1]
                uy = vel[x, y, z, b_idx, 2]
                uz = vel[x, y, z, b_idx, 3]
                rho_local = rho_arr[x, y, z, b_idx]
                d_wall = wall_dist_arr[x, y, z, b_idx]

                nu_mol = (tau_molecular - 0.5f0) / 3.0f0

                # --- Velocity gradients (central differences) ---
                ux_E, uy_E, uz_E = get_velocity_neighbor(vel, x, y, z, b_idx, 1, 0, 0, block_size, neighbor_table)
                ux_W, uy_W, uz_W = get_velocity_neighbor(vel, x, y, z, b_idx, -1, 0, 0, block_size, neighbor_table)
                ux_N, uy_N, uz_N = get_velocity_neighbor(vel, x, y, z, b_idx, 0, 1, 0, block_size, neighbor_table)
                ux_S, uy_S, uz_S = get_velocity_neighbor(vel, x, y, z, b_idx, 0, -1, 0, block_size, neighbor_table)
                ux_T, uy_T, uz_T = get_velocity_neighbor(vel, x, y, z, b_idx, 0, 0, 1, block_size, neighbor_table)
                ux_B, uy_B, uz_B = get_velocity_neighbor(vel, x, y, z, b_idx, 0, 0, -1, block_size, neighbor_table)

                dudx = 0.5f0*(ux_E - ux_W); dudy = 0.5f0*(ux_N - ux_S); dudz = 0.5f0*(ux_T - ux_B)
                dvdx = 0.5f0*(uy_E - uy_W); dvdy = 0.5f0*(uy_N - uy_S); dvdz = 0.5f0*(uy_T - uy_B)
                dwdx = 0.5f0*(uz_E - uz_W); dwdy = 0.5f0*(uz_N - uz_S); dwdz = 0.5f0*(uz_T - uz_B)

                # Strain rate magnitude: |S| = sqrt(2 * Sij * Sij)
                S11 = dudx; S22 = dvdy; S33 = dwdz
                S12 = 0.5f0*(dudy + dvdx)
                S13 = 0.5f0*(dudz + dwdx)
                S23 = 0.5f0*(dvdz + dwdy)
                strain_mag = sqrt(2.0f0 * (S11*S11 + S22*S22 + S33*S33 + 2.0f0*(S12*S12 + S13*S13 + S23*S23)))

                # Vorticity magnitude: |Ω| = sqrt(2 * Ωij * Ωij)
                w_x = dwdy - dvdz
                w_y = dudz - dwdx
                w_z = dvdx - dudy
                vort_mag = sqrt(w_x*w_x + w_y*w_y + w_z*w_z)

                # --- Transition criteria ---
                # Vorticity Reynolds number: Rev = ρ * |S| * d² / μ
                rev = rho_local * strain_mag * d_wall * d_wall / max(nu_mol, 1.0f-10)

                # Turbulent Reynolds number estimate: R_T ≈ ν_t / ν
                # Without a k-equation, approximate from strain/vorticity ratio
                r_t = 0.0f0  # Will be small in laminar regions
                if vort_mag > 1.0f-10 && nu_mol > 1.0f-10
                    # In turbulent regions, strain ≈ vorticity; deviation indicates turbulence
                    r_t = max(strain_mag - vort_mag, 0.0f0) * d_wall * d_wall / (nu_mol * max(vort_mag, 1.0f-10))
                end

                # Onset and length functions
                f_onset = compute_f_onset(rev, re_theta_c, r_t)

                # --- Source terms ---
                # Production: drives γ from 0 → 1
                s_gamma = strain_mag * max(strain_mag, vort_mag)
                s_gamma = sqrt(max(s_gamma, 0.0f0))

                P_gamma = f_length * GAMMA_CA1 * s_gamma * sqrt(max(gamma_local * f_onset, 0.0f0)) * (1.0f0 - GAMMA_CE1 * gamma_local)
                P_gamma = max(P_gamma, 0.0f0)

                # Destruction: relaminarization (active when γ > 1/CE2 and F_turb is small)
                f_turb = compute_f_turb(r_t)
                E_gamma = GAMMA_CA2 * vort_mag * gamma_local * f_turb * (GAMMA_CE2 * gamma_local - 1.0f0)
                E_gamma = max(E_gamma, 0.0f0)

                # --- Advection (1st order upwind) ---
                # ∇γ using upwind differencing based on local velocity
                g_E = get_gamma_neighbor(gamma_in, x, y, z, b_idx, 1, 0, 0, block_size, neighbor_table)
                g_W = get_gamma_neighbor(gamma_in, x, y, z, b_idx, -1, 0, 0, block_size, neighbor_table)
                g_N = get_gamma_neighbor(gamma_in, x, y, z, b_idx, 0, 1, 0, block_size, neighbor_table)
                g_S = get_gamma_neighbor(gamma_in, x, y, z, b_idx, 0, -1, 0, block_size, neighbor_table)
                g_T = get_gamma_neighbor(gamma_in, x, y, z, b_idx, 0, 0, 1, block_size, neighbor_table)
                g_B = get_gamma_neighbor(gamma_in, x, y, z, b_idx, 0, 0, -1, block_size, neighbor_table)

                # Upwind scheme: if u > 0, use backward difference; if u < 0, use forward
                dgdx = ux > 0.0f0 ? (gamma_local - g_W) : (g_E - gamma_local)
                dgdy = uy > 0.0f0 ? (gamma_local - g_S) : (g_N - gamma_local)
                dgdz = uz > 0.0f0 ? (gamma_local - g_B) : (g_T - gamma_local)

                advection = ux * dgdx + uy * dgdy + uz * dgdz

                # --- Diffusion (central differences, Laplacian) ---
                # ∇²γ = Σ (γ_neighbor - 2*γ_center + γ_opposite) / Δx²
                # In lattice units Δx = 1
                diffusion = (g_E + g_W + g_N + g_S + g_T + g_B - 6.0f0 * gamma_local)

                # Effective diffusion coefficient: ν_mol / σ_γ + small constant
                D_eff = nu_mol / GAMMA_SIGMA + gamma_diffusion

                # --- Time integration (explicit Euler, Δt = 1 in lattice units) ---
                gamma_new = gamma_local - advection + D_eff * diffusion + (P_gamma - E_gamma)

                # Clamp to [0, 1]
                gamma_new = min(max(gamma_new, 0.0f0), 1.0f0)

                gamma_out[x, y, z, b_idx] = gamma_new
            end
        end
    end
end

# ==============================================================================
# HIGH-LEVEL INTERFACE
# ==============================================================================

"""
Perform one gamma-transport step for a given grid level.

Must be called AFTER the main LBM stream-collide so that velocity/density
fields are up to date.
"""
function perform_gamma_step!(
    level,
    gamma_out, gamma_in, vel_field,
    tau_molecular::Float32,
    re_theta_c::Float32,
    f_length::Float32,
    tu_intensity::Float32,
    gamma_diffusion::Float32
)
    backend = get_backend(gamma_in)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0; return; end

    kernel! = gamma_transport_kernel!(backend)
    kernel!(
        gamma_out, gamma_in,
        vel_field, level.rho, level.obstacle,
        level.wall_dist, level.neighbor_table,
        level.map_x, level.map_y, level.map_z,
        tau_molecular,
        re_theta_c,
        f_length,
        tu_intensity,
        gamma_diffusion,
        Int32(n_blocks),
        Int32(BLOCK_SIZE),
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )

    KernelAbstractions.synchronize(backend)
end
