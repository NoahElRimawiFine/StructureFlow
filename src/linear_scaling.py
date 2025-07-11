import time
import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os

import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint
from scipy.sparse import spmatrix
import ot

import fm
from models.components.base import MLPODEF
from geomloss import SamplesLoss

CUDA_DEVICE = int(os.environ.get("CUDA_DEVICE", 1))
if not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{CUDA_DEVICE}")
    print(f"Using CUDA device {CUDA_DEVICE}")

torch.set_default_dtype(torch.float32)


def prepare_time_binned_data(adata, time_column="t"):
    num_time_bins = adata.obs[time_column].nunique()
    time_bins = sorted(adata.obs[time_column].unique())
    grouped_data = []
    for t_bin in time_bins:
        cells_t = adata[adata.obs[time_column] == t_bin].X
        if isinstance(cells_t, spmatrix):
            cells_t = cells_t.toarray()
        grouped_data.append(torch.from_numpy(cells_t).float())
    return grouped_data


def normalize_data(grouped_data):
    all_cells = torch.cat(grouped_data, dim=0)
    scaler = StandardScaler()
    all_cells_np = all_cells.numpy()
    scaler.fit(all_cells_np)
    normalized_data = []
    for tensor in grouped_data:
        normalized = torch.from_numpy(scaler.transform(tensor.numpy())).to(
            torch.float32
        )
        normalized_data.append(normalized)
    return normalized_data, scaler


def proximal(w, dims, lam=0.1, eta=0.1):
    wadj = w.view(dims[0], -1, dims[0])
    tmp = torch.sum(wadj**2, dim=1).pow(0.5) - lam * eta
    alpha = torch.clamp(tmp, min=0)
    v = torch.nn.functional.normalize(wadj, dim=1) * alpha[:, None, :]
    w.data = v.view(-1, dims[0])


def generate_linear_system(
    n_nodes,
    N_samples_in=500,
    T_timepoints=5,
    t1_end_time=2.5,
    dt_val=0.001,
    p_edge=0.25,
    rand_seed=42,
):
    effective_dt = dt_val
    if n_nodes >= 500:
        effective_dt = 1e-5
        print(
            f"Using smaller dt={effective_dt} for n={n_nodes} in generate_linear_system"
        )
    elif n_nodes >= 100:
        effective_dt = 1e-4
        print(
            f"Using smaller dt={effective_dt} for n={n_nodes} in generate_linear_system"
        )
    ts = np.linspace(0, t1_end_time, T_timepoints)
    np.random.seed(rand_seed)
    G = nx.erdos_renyi_graph(n_nodes, p_edge, directed=True, seed=rand_seed)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.choice([-1, 1])
    A_orig = nx.to_numpy_array(G, weight="weight")
    max_real_eig_orig = np.max(np.real(np.linalg.eigvals(A_orig)))
    stabilizing_shift = max_real_eig_orig + 0.1
    A_true_unscaled = -(A_orig - stabilizing_shift * np.eye(n_nodes))
    if max_real_eig_orig > 10:
        print(
            f"Original max real eigenvalue ({max_real_eig_orig:.2f}) was large. Scaling A_true."
        )
        A_true = A_true_unscaled / stabilizing_shift
    else:
        A_true = A_true_unscaled
    print(
        f"Max real eigenvalue of final A_true: {np.max(np.real(np.linalg.eigvals(A_true)))}"
    )
    d_vars = A_true.shape[0]
    sigma_val = 0.1
    noise_sigma_mat = np.eye(d_vars) * sigma_val

    def simulate(A_mat, N_sim_samples):
        xs_data = []
        for i_tp in range(T_timepoints):
            x = np.random.randn(N_sim_samples, d_vars) * 0.1 + 0.25
            t_sim = 0
            while t_sim < ts[i_tp]:
                if not np.all(np.isfinite(x)):
                    print(
                        f"Warning: Non-finite values in x before update at t_sim={t_sim}, i_tp={i_tp}. Clipping."
                    )
                    x = np.clip(x, -1e8, 1e8)
                noise_term = (
                    np.random.randn(N_sim_samples, d_vars)
                    @ noise_sigma_mat
                    * effective_dt**0.5
                )
                dynamics_term = -x @ A_mat.T * effective_dt
                x_new = x + dynamics_term + noise_term
                if not np.all(np.isfinite(x_new)):
                    print(
                        f"Warning: Non-finite values in x_new after update at t_sim={t_sim}, i_tp={i_tp}. Clipping x_new."
                    )
                    x = np.clip(x_new, -1e8, 1e8)
                else:
                    x = x_new
                t_sim += effective_dt
            xs_data.append(x)
        return np.stack(xs_data)

    ko_idx_list = []
    Ms_perturb = [
        np.ones(
            (d_vars, d_vars),
        )
    ]
    ko_label_list = [
        "wt",
    ]
    for i_ko in ko_idx_list:
        M_perturb_mat = np.ones((d_vars, d_vars))
        M_perturb_mat[i_ko, :] = 0
        Ms_perturb.append(M_perturb_mat)
        ko_label_list.append(str(i_ko))
    xs_simulated = []
    for M_perturb_item in Ms_perturb:
        xs_simulated.append(
            simulate(A_true * M_perturb_item, int(N_samples_in // len(Ms_perturb)))
        )
    xs_simulated = np.stack(xs_simulated)
    return xs_simulated, A_true, noise_sigma_mat


def train_random_discrete_time_ode(
    func,
    normalized_data,
    A_true_matrix,
    batch_size_ode,
    n_steps_per_transition=1000,
    plot_freq=1000,
    l1_reg=0.0,
    l2_reg=0.05,
    alpha_wass=0.1,
    plot=False,
    device=DEVICE,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    horizon = 1
    num_time_bins = len(normalized_data)
    num_transitions = num_time_bins - 1
    transition_times = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)
    sinkhorn_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="auto")

    def create_batch_for_transition(t_start_idx, batch_size_val):
        cells_t0 = normalized_data[t_start_idx]
        cells_t1 = normalized_data[t_start_idx + horizon]
        indices_t0 = torch.randint(0, cells_t0.shape[0], (batch_size_val,))
        indices_t1 = torch.randint(0, cells_t1.shape[0], (batch_size_val,))
        x0_batch = cells_t0[indices_t0].to(device)
        x1_batch = cells_t1[indices_t1].to(device)
        return x0_batch, transition_times, x1_batch

    lr_val = 0.005
    optimizer = torch.optim.Adam(func.parameters(), lr=lr_val)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    total_steps = n_steps_per_transition * num_transitions
    func = func.to(device)

    ode_integration_time_start = time.time()
    for step in tqdm(range(1, total_steps + 1)):
        transition_idx = np.random.randint(0, num_transitions)
        t_start_val = transition_idx
        x0_ode, t_ode, x1_ode = create_batch_for_transition(t_start_val, batch_size_ode)
        x0_ode = x0_ode.unsqueeze(1)
        z0_ode = x0_ode
        z_pred: torch.Tensor = odeint(func, z0_ode, t_ode)  # type: ignore[assignment]
        z_pred = z_pred[-1].squeeze(1)
        loss = sinkhorn_loss_fn(z_pred, x1_ode)
        if l2_reg != 0:
            loss += l2_reg * func.l2_reg()
        if l1_reg != 0:
            loss += l1_reg * func.fc1_reg()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proximal(func.fc1.weight, func.dims, lam=func.GL_reg, eta=0.01)
        scheduler.step()
    ode_integration_time_only = time.time() - ode_integration_time_start

    with torch.no_grad():
        W_v_node = func.causal_graph(w_threshold=0.0)
        y_true_flat = np.abs(np.sign(A_true_matrix).astype(int).flatten())
    return None, W_v_node, y_true_flat, ode_integration_time_only


def jacobian_inference(A_estim_mat, y_true_vec):
    if A_estim_mat is None or A_estim_mat.size == 0:
        return None, None, None, np.nan
    y_pred_vec = np.abs(A_estim_mat.flatten())
    try:
        auroc_val = roc_auc_score(y_true_vec, y_pred_vec)
    except ValueError as e:
        print(
            f"Could not calculate AUROC for Jacobian: {e}. y_true_vec may have only one class."
        )
        auroc_val = np.nan
    return None, None, None, auroc_val


def causal_graph_extraction(W_v_mat, y_true_vec):
    y_pred_mlp_vec = np.abs(W_v_mat.flatten())
    try:
        auroc_val = roc_auc_score(y_true_vec, y_pred_mlp_vec)
    except ValueError as e:
        print(
            f"Could not calculate AUROC for causal graph: {e}. y_true_vec may have only one class."
        )
        auroc_val = np.nan
    return None, None, None, auroc_val


def _compute_jacobian_manually_chunked(func_to_diff, primary_input, chunk_size_cols=16):
    device = primary_input.device
    input_leaf = primary_input.detach().requires_grad_(True)
    output_y_raw = func_to_diff(input_leaf)
    if output_y_raw.ndim == 0:
        output_y_for_grad = output_y_raw.unsqueeze(0)
        actual_n_out = 1
    elif output_y_raw.ndim == 1:
        output_y_for_grad = output_y_raw
        actual_n_out = output_y_raw.shape[0]
    elif output_y_raw.ndim == 2 and output_y_raw.shape[0] == 1:
        output_y_for_grad = output_y_raw.squeeze(0)
        actual_n_out = output_y_raw.shape[1]
    elif output_y_raw.ndim == 2 and output_y_raw.shape[1] == 1:
        output_y_for_grad = output_y_raw.squeeze(1)
        actual_n_out = output_y_raw.shape[0]
    else:
        raise ValueError(
            f"func_to_diff output has unhandled shape: {output_y_raw.shape}. Expected effectively 1D."
        )
    n_in_dims = input_leaf.shape[0]
    jacobian_mat = torch.zeros(
        (actual_n_out, n_in_dims), device=device, dtype=output_y_for_grad.dtype
    )
    for j_start in range(0, actual_n_out, chunk_size_cols):
        j_end = min(j_start + chunk_size_cols, actual_n_out)
        for k_abs_val in range(j_start, j_end):
            v_basis = torch.zeros(
                actual_n_out, device=device, dtype=output_y_for_grad.dtype
            )
            v_basis[k_abs_val] = 1.0
            if input_leaf.grad is not None:
                input_leaf.grad.zero_()
            (grad_input_val,) = torch.autograd.grad(
                outputs=output_y_for_grad,
                inputs=input_leaf,
                grad_outputs=v_basis,
                retain_graph=True,
            )
            jacobian_mat[k_abs_val, :] = grad_input_val
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return jacobian_mat


def train_ngm_sf2m(
    A_true_matrix,
    xs_all_conditions,
    n_vars,
    sigma_matrix,
    N_samples_per_tp_gen=500,
    T_timepoints_gen=5,
    t1_end_time_gen=2.5,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dims_sf = [n_vars, 64, 64, 1]
    s_model_fm = fm.MLP(d=n_vars, hidden_sizes=[64, 64], time_varying=True)
    v_model_fm = MLPODEF(dims=dims_sf, GL_reg=0.01, bias=True, time_invariant=True)
    s_model_fm = s_model_fm.to(DEVICE)
    v_model_fm = v_model_fm.to(DEVICE)
    optim_fm = torch.optim.AdamW(
        list(s_model_fm.parameters()) + list(v_model_fm.parameters()), 3e-3
    )
    xs_wt_unnormalized_np = xs_all_conditions[0]
    num_timepoints_from_data = xs_wt_unnormalized_np.shape[0]
    grouped_data_wt = [
        torch.from_numpy(xs_wt_unnormalized_np[t_idx]).float()
        for t_idx in range(num_timepoints_from_data)
    ]
    normalized_grouped_data_wt, _ = normalize_data(grouped_data_wt)
    normalized_cells_wt_all_times = torch.cat(normalized_grouped_data_wt, dim=0).to(
        DEVICE
    )

    class StableBridgeMatcher(fm.BridgeMatcher):
        def sample_map(self, pi_coupling, batch_size_val, replace=True):
            pi_coupling = pi_coupling.to(DEVICE)
            p_flat = pi_coupling.flatten()
            p_flat = torch.abs(p_flat)
            p_flat = torch.nan_to_num(p_flat, nan=1e-10, posinf=1.0, neginf=1e-10)
            p_flat = torch.clip(p_flat, min=1e-10)
            p_sum_val = p_flat.sum()
            if p_sum_val == 0:
                p_flat = torch.ones_like(p_flat, device=DEVICE)
            p_flat = p_flat / p_flat.sum()
            for attempt in range(3):
                try:
                    if torch.isnan(p_flat).any() or torch.isinf(p_flat).any():
                        raise RuntimeError("Invalid probability distribution detected")
                    p_flat = p_flat / p_flat.sum()
                    choices_idx = torch.multinomial(
                        p_flat, num_samples=batch_size_val, replacement=replace
                    )
                    choices_np_arr = choices_idx.detach().cpu().numpy()
                    pi_shape_1 = (
                        pi_coupling.shape[1]
                        if isinstance(pi_coupling.shape[1], int)
                        else pi_coupling.shape[1].cpu().item()
                    )
                    i_indices, j_indices = np.divmod(choices_np_arr, pi_shape_1)
                    return i_indices, j_indices
                except RuntimeError as e_runtime:
                    if (
                        "invalid multinomial distribution" in str(e_runtime)
                        and attempt < 2
                    ):
                        min_clip_val = 1e-8 * (10 ** (attempt + 1))
                        p_flat = torch.clip(p_flat, min=min_clip_val)
                        p_flat = p_flat / p_flat.sum()
                        print(
                            f"Retrying sample_map with stronger stabilization: min_clip={min_clip_val}"
                        )
                    else:
                        print(
                            "Falling back to uniform sampling due to persistent numerical issues"
                        )
                        num_elements_p = p_flat.shape[0]
                        uniform_choices_idx = torch.randint(
                            0, num_elements_p, (batch_size_val,), device=DEVICE
                        )
                        choices_np_fallback = uniform_choices_idx.cpu().numpy()
                        i_indices_fb, j_indices_fb = np.divmod(
                            choices_np_fallback, pi_shape_1
                        )
                        return i_indices_fb, j_indices_fb

    def create_stable_entropic_otfm(
        x_samples_norm, n_dim, noise_mat, N_samples_tp, T_intervals, t1_total_time
    ):
        epsilon_scale_val = 4000.0 * n_dim
        scalar_sigma_fm = noise_mat[0, 0]
        if isinstance(scalar_sigma_fm, torch.Tensor):
            scalar_sigma_fm = scalar_sigma_fm.item()
        elif not isinstance(scalar_sigma_fm, (float, int)):
            try:
                scalar_sigma_fm = float(scalar_sigma_fm)
            except TypeError:
                raise TypeError(
                    f"Could not convert extracted sigma {scalar_sigma_fm} to float for EntropicOTFM"
                )
        ot_model = fm.EntropicOTFM(
            x_samples_norm,
            torch.tile(
                torch.arange(T_intervals, device=DEVICE), (N_samples_tp, 1)
            ).T.reshape(-1),
            dt=t1_total_time / T_intervals,
            sigma=scalar_sigma_fm,
            T=T_intervals,
            dim=n_dim,
            device=DEVICE,
        )
        ot_model.bm = StableBridgeMatcher()
        for i_interval in range(ot_model.T - 1):
            x0_ot = ot_model.x[ot_model.t_idx == i_interval, :].to(DEVICE)
            x1_ot = ot_model.x[ot_model.t_idx == i_interval + 1, :].to(DEVICE)
            x0_cpu_ot = x0_ot.cpu().numpy()
            x1_cpu_ot = x1_ot.cpu().numpy()
            max_z_val_clip = 20.0
            x0_cpu_ot = np.nan_to_num(
                x0_cpu_ot, nan=0.0, posinf=max_z_val_clip, neginf=-max_z_val_clip
            )
            x1_cpu_ot = np.nan_to_num(
                x1_cpu_ot, nan=0.0, posinf=max_z_val_clip, neginf=-max_z_val_clip
            )
            x0_cpu_ot = np.clip(x0_cpu_ot, -max_z_val_clip, max_z_val_clip)
            x1_cpu_ot = np.clip(x1_cpu_ot, -max_z_val_clip, max_z_val_clip)
            if x0_cpu_ot.shape[0] == 0 or x1_cpu_ot.shape[0] == 0:
                print(
                    f"Warning: Empty input arrays for Sinkhorn at time step {i_interval}. Using fallback uniform plan."
                )
                n_x0_fb_val = max(1, x0_cpu_ot.shape[0])
                n_x1_fb_val = max(1, x1_cpu_ot.shape[0])
                ot_model.Ts[i_interval] = torch.ones(
                    n_x0_fb_val, n_x1_fb_val, device=DEVICE
                ) / (n_x0_fb_val * n_x1_fb_val)
                continue
            C_cost_mat = (
                ot.utils.euclidean_distances(x0_cpu_ot, x1_cpu_ot, squared=True) / 2
            )
            C_cost_mat = np.nan_to_num(C_cost_mat, nan=0.0, posinf=1e6, neginf=0.0)
            C_cost_mat = np.clip(C_cost_mat, 0, 1e6)
            p_marginal_np = (
                torch.ones(x0_cpu_ot.shape[0], dtype=torch.float64) / x0_cpu_ot.shape[0]
            ).numpy()
            q_marginal_np = (
                torch.ones(x1_cpu_ot.shape[0], dtype=torch.float64) / x1_cpu_ot.shape[0]
            ).numpy()
            C_cost_np_float64 = C_cost_mat.astype(np.float64)
            eps_reg = epsilon_scale_val * ot_model.dt * ot_model.sigma**2
            try:
                plan_ot = ot.sinkhorn(
                    p_marginal_np,
                    q_marginal_np,
                    C_cost_np_float64,
                    eps_reg,
                    method="sinkhorn_stabilized",
                    numItermax=2000,
                    stopThr=1e-5,
                )
                ot_model.Ts[i_interval] = torch.tensor(
                    plan_ot, device=DEVICE, dtype=torch.float32
                )
            except Exception as e_sinkhorn:
                print(
                    f"Sinkhorn failed at time {i_interval}, using approximate plan: {e_sinkhorn}"
                )
                n_x0_err, n_x1_err = x0_ot.shape[0], x1_ot.shape[0]
                ot_model.Ts[i_interval] = torch.ones(
                    n_x0_err, n_x1_err, device=DEVICE
                ) / (n_x0_err * n_x1_err)
        return ot_model

    print("Generating stabilized entropic OT model...")
    otfm_model = create_stable_entropic_otfm(
        normalized_cells_wt_all_times,
        n_vars,
        sigma_matrix,
        N_samples_per_tp_gen,
        T_timepoints_gen,
        t1_end_time_gen,
    )
    print("Model created successfully!")

    trace_loss = []
    alpha_fm = 0.1
    reg_fm = 1e-5
    training_steps_fm = 4_000
    batch_size_fm = 64

    pure_training_start_time = time.time()

    for i_step in tqdm(range(training_steps_fm)):
        try:
            _x_batch, _s_batch, _u_batch, _t_batch, _t_orig_batch = (
                otfm_model.sample_bridges_flows(batch_size=batch_size_fm)
            )
            _x_batch, _s_batch, _u_batch, _t_batch, _t_orig_batch = (
                _val.to(DEVICE)
                for _val in [_x_batch, _s_batch, _u_batch, _t_batch, _t_orig_batch]
            )
            if (
                torch.isnan(_x_batch).any()
                or torch.isnan(_s_batch).any()
                or torch.isnan(_u_batch).any()
                or torch.isinf(_x_batch).any()
            ):
                print(
                    f"Warning: NaN/Inf values detected at iteration {i_step}. Skipping this batch."
                )
                continue
            optim_fm.zero_grad()
            s_input_fm = _x_batch.unsqueeze(1)
            v_input_fm = _x_batch.unsqueeze(1)
            s_fit_fm = s_model_fm(_t_batch, _x_batch).squeeze(1)
            v_fit_fm = v_model_fm(_t_batch, v_input_fm).squeeze(
                1
            ) - otfm_model.sigma**2 / 2 * s_model_fm(_t_batch, _x_batch)
            L_score_fm = torch.mean(
                (_t_orig_batch * (1 - _t_orig_batch)) * (s_fit_fm - _s_batch) ** 2
            )
            L_flow_fm = torch.mean((v_fit_fm * otfm_model.dt - _u_batch) ** 2)
            L_reg_fm = v_model_fm.l2_reg() + v_model_fm.fc1_reg()
            L_total_fm = (
                alpha_fm * L_score_fm + (1 - alpha_fm) * L_flow_fm + reg_fm * L_reg_fm
            )
            if i_step % 500 == 0:
                print(L_score_fm.item(), L_flow_fm.item(), L_reg_fm.item())
            trace_loss.append(L_total_fm.item())
            L_total_fm.backward()
            optim_fm.step()
            proximal(
                v_model_fm.fc1.weight, v_model_fm.dims, lam=v_model_fm.GL_reg, eta=0.01
            )
        except RuntimeError as e_runtime_train:
            if "invalid multinomial distribution" in str(e_runtime_train):
                print(
                    f"Numerical instability (multinomial) at iter {i_step}. Reducing batch size."
                )
                batch_size_fm = max(16, batch_size_fm // 2)
                print(f"Reduced batch size to {batch_size_fm}")
                continue
            else:
                raise

    pure_training_elapsed_time = time.time() - pure_training_start_time
    print(f"Pure Flow Matching training loop time: {pure_training_elapsed_time:.2f}s")

    def get_flow_local_fm(t_scalar, x_vector_in):
        t_scalar_dev = t_scalar.to(DEVICE)
        x_vector_dev = x_vector_in.to(DEVICE)
        x_input_for_v = x_vector_dev.view(1, 1, -1)
        t_input_for_v = t_scalar_dev.view(1, 1, 1)
        res_flow = v_model_fm(t_input_for_v, x_input_for_v)
        return res_flow.squeeze(0)

    J_list_time_means_fm = []
    for i_time_jac in range(otfm_model.T):
        t_current_jac = torch.tensor(i_time_jac * otfm_model.dt, device=DEVICE)
        samples_at_t_current_jac = otfm_model.x[otfm_model.t_idx == i_time_jac, :].to(
            DEVICE
        )
        if samples_at_t_current_jac.shape[0] == 0:
            print(
                f"Warning: No samples found for time index {i_time_jac} in Jacobian calculation. Skipping."
            )
            continue
        J_sum_for_current_t_fm = torch.zeros(
            (n_vars, n_vars), device=DEVICE, dtype=torch.float32
        )
        num_samples_processed_for_t_fm = 0
        for sample_idx_jac in range(samples_at_t_current_jac.shape[0]):
            x_sample_jac = samples_at_t_current_jac[sample_idx_jac, :]
            func_to_diff_fm = lambda x_arg: get_flow_local_fm(t_current_jac, x_arg)
            J_single_sample_fm = _compute_jacobian_manually_chunked(
                func_to_diff_fm,
                x_sample_jac,
                chunk_size_cols=max(1, n_vars // 32 if n_vars > 32 else 1),
            )
            J_sum_for_current_t_fm += J_single_sample_fm
            num_samples_processed_for_t_fm += 1
        if num_samples_processed_for_t_fm > 0:
            J_mean_for_current_t_fm = (
                J_sum_for_current_t_fm / num_samples_processed_for_t_fm
            )
            J_list_time_means_fm.append(J_mean_for_current_t_fm)
        else:
            print(
                f"Warning: No Jacobians computed for time index {i_time_jac} despite having samples. Skipping."
            )
    if not J_list_time_means_fm:
        print(
            "Warning: No Jacobians computed across any time points. Returning zero matrix for A_estim."
        )
        A_estim_np_fm = np.zeros((n_vars, n_vars), dtype=np.float32)
    else:
        A_estim_tensor_fm = -torch.stack(J_list_time_means_fm).mean(dim=0)
        A_estim_np_fm = A_estim_tensor_fm.cpu().numpy()
    with torch.no_grad():
        W_v_fc1 = v_model_fm.causal_graph(w_threshold=0.0)
    y_true_flat_fm = np.abs(np.sign(A_true_matrix).astype(int).flatten())
    jacobian_derived_interaction_matrix = A_estim_np_fm
    direct_fc1_causal_graph = W_v_fc1

    return (
        jacobian_derived_interaction_matrix,
        direct_fc1_causal_graph,
        y_true_flat_fm,
        pure_training_elapsed_time,
    )


def run_experiment_for_n(n_val_in, seed_val):
    print(f"\nRunning experiment for n = {n_val_in}, seed = {seed_val}...")
    xs_raw_data, A_true_data, noise_matrix_sigma_data = generate_linear_system(
        n_val_in, rand_seed=seed_val
    )
    print(
        f"Shape of xs_raw before cleaning: {xs_raw_data.shape if xs_raw_data is not None else 'None'}"
    )
    xs_finite_data = np.nan_to_num(xs_raw_data, nan=0.0, posinf=1e6, neginf=-1e6)
    xs_clipped_data = np.clip(xs_finite_data, -1e3, 1e3)
    print(
        f"Data range in xs_clipped: min={np.min(xs_clipped_data)}, max={np.max(xs_clipped_data)}"
    )
    xs_processed_data = xs_clipped_data

    node_total_time = np.nan
    auroc_disc_causal_val = np.nan
    node_ode_integration_time_specific = np.nan

    if n_val_in <= 100:
        print(f"Running NGM-NODE for n={n_val_in}")

        node_start_time = time.time()

        hidden_dim_node = min(200, max(50, n_val_in * 2))
        print(f"Using hidden dimension: {hidden_dim_node}")

        dims_node_val = [n_val_in, hidden_dim_node, 1]
        func_node_model = MLPODEF(dims=dims_node_val, GL_reg=0.05).to(DEVICE)

        xs_wt_node_processed_data = xs_processed_data[0]
        grouped_data_node_list = [
            torch.from_numpy(xs_wt_node_processed_data[t_idx]).float()
            for t_idx in range(xs_wt_node_processed_data.shape[0])
        ]
        normalized_data_node_list, _ = normalize_data(grouped_data_node_list)

        n_steps = 1000
        batch_size_ode_for_n = 64

        _, W_v_disc_val, y_true_disc_val, node_ode_integration_time_specific = (
            train_random_discrete_time_ode(
                func_node_model,
                normalized_data_node_list,
                A_true_data,
                batch_size_ode=batch_size_ode_for_n,
                n_steps_per_transition=n_steps,
                device=DEVICE,
                seed=seed_val,
            )
        )

        node_total_time = time.time() - node_start_time
        print(f"Total NGM-NODE block time: {node_total_time:.2f}s")
        print(
            f"NGM-NODE ODE integration loop time: {node_ode_integration_time_specific:.2f}s"
        )

        _, _, _, auroc_disc_causal_val = causal_graph_extraction(
            W_v_disc_val, y_true_disc_val
        )
    else:
        print(f"Skipping NGM-NODE for n={n_val_in} (cap is 100)")

    print(f"Running StructureFlow for n={n_val_in}")
    _, direct_graph_ngm_val, y_true_ngm_val, pure_train_time_sf = train_ngm_sf2m(
        A_true_data, xs_processed_data, n_val_in, noise_matrix_sigma_data, seed=seed_val
    )

    _, _, _, auroc_ngm_causal_val = causal_graph_extraction(
        direct_graph_ngm_val, y_true_ngm_val
    )

    return {
        "StructureFlow Pure Training Time (s)": pure_train_time_sf,
        "NGM-NODE Time (s)": node_total_time,
        "NGM-NODE ODE Integration Time (s)": node_ode_integration_time_specific,
        "StructureFlow Causal Graph AUROC": auroc_ngm_causal_val,
        "NGM-NODE Causal Graph AUROC": auroc_disc_causal_val,
    }


def main():

    n_val_str = os.environ.get("SYSTEM_SIZE_N")
    seed_str = os.environ.get("EXPERIMENT_SEED")

    if n_val_str is None:
        print("Error: Environment variable SYSTEM_SIZE_N is not set.")
        print("Please set to desired system size (number of variables).")
        return
    try:
        n_val_run = int(n_val_str)
    except ValueError:
        print(f"Error: SYSTEM_SIZE_N ('{n_val_str}') is not a valid integer.")
        return

    if seed_str is None:
        print("Error: Environment variable EXPERIMENT_SEED is not set.")
        print("Please set it to the desired random seed.")
        return
    try:
        experiment_seed_run = int(seed_str)
    except ValueError:
        print(f"Error: EXPERIMENT_SEED ('{seed_str}') is not a valid integer.")
        return

    print(f"\nStarting experiment for n = {n_val_run}, seed = {experiment_seed_run}")
    result_dict_run = run_experiment_for_n(n_val_run, experiment_seed_run)

    row_data_df = {"Number of Variables (n)": n_val_run, "Seed": experiment_seed_run}
    row_data_df.update(result_dict_run)
    detailed_df_data = pd.DataFrame([row_data_df])

    csv_filename_out = f"results_n{n_val_run}_seed_{experiment_seed_run}.csv"
    detailed_df_data.to_csv(csv_filename_out, index=False)
    print(f"Saved results to '{csv_filename_out}'.")

    summary_filename_out = (
        f"results_summary_n{n_val_run}_seed_{experiment_seed_run}.txt"
    )
    summary_lines_list = [
        f"Summary of Results for n={n_val_run}, Seed {experiment_seed_run}\n",
        "=======================================================\n\n",
    ]
    metrics = [
        "StructureFlow Pure Training Time (s)",
        "NGM-NODE Time (s)",
        "NGM-NODE ODE Integration Time (s)",
        "StructureFlow Causal Graph AUROC",
        "NGM-NODE Causal Graph AUROC",
    ]
    for metric_key in metrics:
        if metric_key in result_dict_run:
            metric_val = result_dict_run[metric_key]
            val_str_display = (
                f"{metric_val:.3f}"
                if isinstance(metric_val, (float, int)) and not np.isnan(metric_val)
                else "N/A"
            )
            summary_lines_list.append(f"  {metric_key}: {val_str_display}\n")
    summary_lines_list.append("\n")

    with open(summary_filename_out, "w") as f_txt_out:
        f_txt_out.writelines(summary_lines_list)
    print(f"Saved text summary to '{summary_filename_out}'.")


if __name__ == "__main__":
    main()
