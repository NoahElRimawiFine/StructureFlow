import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text
from tqdm import tqdm
import pandas as pd

import torch
from torch._functorch.eager_transforms import jacrev
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint
import scipy as sp

import fm
from models.components.base import MLPODEF
from geomloss import SamplesLoss

DEVICE = torch.device("mps")
torch.set_default_dtype(torch.float32)

def prepare_time_binned_data(adata, time_column='t'):
    """
    Groups cells by their time bins and returns a list of tensors.
    
    Args:
        adata (AnnData): The AnnData object containing cell data.
        time_column (str): The column in adata.obs indicating time bins.
        
    Returns:
        List[torch.Tensor]: A list where each element is a tensor of cells at a specific time bin.
    """
    num_time_bins = adata.obs[time_column].nunique()
    time_bins = sorted(adata.obs[time_column].unique())
    grouped_data = []
    for t in time_bins:
        cells_t = adata[adata.obs[time_column] == t].X
        if isinstance(cells_t, sp.spmatrix):
            cells_t = cells_t.toarray()
        grouped_data.append(torch.from_numpy(cells_t).float())
    return grouped_data


def normalize_data(grouped_data):
    """
    Applies Z-score normalization to each gene across all cells.
    
    Args:
        grouped_data (List[torch.Tensor]): List of tensors grouped by time bins.
        
    Returns:
        List[torch.Tensor]: Normalized data.
    """
    all_cells = torch.cat(grouped_data, dim=0)
    scaler = StandardScaler()
    all_cells_np = all_cells.numpy()
    scaler.fit(all_cells_np)

    normalized_data = []
    for tensor in grouped_data:
        normalized = torch.from_numpy(scaler.transform(tensor.numpy())).to(torch.float32)
        normalized_data.append(normalized)
    return normalized_data, scaler


def proximal(w, dims, lam=0.1, eta=0.1):
    """Proximal step for group sparsity"""
    # w shape [j * m1, i]
    wadj = w.view(dims[0], -1, dims[0])  # [j, m1, i]
    tmp = torch.sum(wadj**2, dim=1).pow(0.5) - lam * eta
    alpha = torch.clamp(tmp, min=0)
    v = torch.nn.functional.normalize(wadj, dim=1) * alpha[:, None, :]
    w.data = v.view(-1, dims[0])


# Generates Stephen's linear system for "n" nodes
def generate_linear_system(n, N=500, T=5, t1=2.5, dt=0.001, p=0.25, rand_seed=42):

    ts = np.linspace(0, t1, T)
    # Parameters
    np.random.seed(rand_seed)
    # Generate Erdős-Rényi graph
    G = nx.erdos_renyi_graph(n, p, directed=True, seed=42)
    # Assign random signed weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.choice([-1, 1])

    # Convert to adjacency matrix
    A_true = nx.to_numpy_array(G, weight="weight")
    A_true = -(
        A_true - (np.max(np.real(np.linalg.eig(A_true)[0])) + 1e-2) * np.eye(n)
    )  # Hurwitz stable

    d = A_true.shape[0]
    sigma = np.eye(d) * 0.1

    def simulate(A, N):
        xs = []
        for i in range(T):
            x = np.random.randn(N, d) * 0.1 + 0.25
            t = 0
            while t < ts[i]:
                x += -x @ A.T * dt + np.random.randn(N, d) @ sigma * dt**0.5
                t += dt
            xs.append(x)
        return np.stack(xs)

    # knockouts
    # ko_idx = np.arange(d)
    # ko_idx = [9]
    ko_idx = []
    Ms = [
        np.ones(
            (d, d),
        ),
    ]
    ko_label = [
        "wt",
    ]
    for i in ko_idx:
        M = np.ones((d, d))
        M[i, :] = 0
        Ms.append(M)
        ko_label.append(str(i))

    xs = []
    for M in Ms:
        xs.append(simulate(A_true * M, int(N // len(Ms))))
    xs = np.stack(xs)

    return xs, A_true, sigma


def train_random_discrete_time_ode(
    func,
    normalized_data,
    A_true,  # added argument to compute y_true later
    n_steps_per_transition=1000,
    plot_freq=1000,
    l1_reg=0.0,
    l2_reg=0.05,
    alpha_wass=0.1,
    plot=False,
    device=DEVICE,
):
    """
    Train Neural ODE on discrete, randomly binned time points.
    (Modified: No plotting inside the function. At the end, we extract:
       - A_estim: average Jacobian from a representative batch,
       - W_v: causal graph extracted from the hidden layer weights,
       - y_true: baseline from A_true.)
    """
    horizon = 1  # transition from time ti to ti+1
    num_time_bins = len(normalized_data)
    num_transitions = num_time_bins - 1
    num_variables = normalized_data[0].shape[1]
    transition_times = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)
    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="auto")

    def create_batch_for_transition(t_start, batch_size):
        cells_t0 = normalized_data[t_start]
        cells_t1 = normalized_data[t_start + horizon]
        indices_t0 = torch.randint(0, cells_t0.shape[0], (batch_size,))
        indices_t1 = torch.randint(0, cells_t1.shape[0], (batch_size,))
        x0 = cells_t0[indices_t0].to(device)
        x1 = cells_t1[indices_t1].to(device)
        return x0, transition_times, x1

    lr = 0.005
    optimizer = torch.optim.Adam(func.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    total_steps = n_steps_per_transition * num_transitions

    # Training loop (no plotting inside)
    for step in tqdm(range(1, total_steps + 1)):
        transition_idx = np.random.randint(0, num_transitions)
        t_start = transition_idx
        batch_size = 20
        x0, t, x1 = create_batch_for_transition(t_start, batch_size)
        x0 = x0.unsqueeze(1)
        z0 = x0

        # Integrate ODE on CPU, faster than MPS
        z0_cpu = z0.cpu()
        t_cpu = t.cpu()
        func_cpu = func.cpu()
        z_pred_cpu = odeint(func_cpu, z0_cpu, t_cpu)
        z_pred = z_pred_cpu.to(device)
        z_pred = z_pred[-1].squeeze(1)

        # Train on GPU (MPS)
        loss = sinkhorn_loss(z_pred, x1)
        if l2_reg != 0:
            loss += l2_reg * func.l2_reg()
        if l1_reg != 0:
            loss += l1_reg * func.fc1_reg()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proximal(func.fc1.weight, func.dims, lam=func.GL_reg, eta=0.01)
        scheduler.step()

    # After training, extract metrics from the trained discrete-time ODE model.
    with torch.no_grad():
        func_cpu = func.cpu()
        # Because MLPODEF's forward is time invariant (or nearly so), we use a dummy time input.
        dummy_t = torch.tensor([[0.0]], dtype=torch.float32)
        # Use a representative batch: all cells from the first time bin.
        sample_x = normalized_data[0].to("cpu")  # shape: [N, num_variables]
        # Define a function that maps a single x to the output:
        def get_flow(x):
            # x: [num_variables] -> output: [num_variables]
            return func_cpu(dummy_t, x.unsqueeze(0)).squeeze(0)
        # Compute Jacobian for each x using functorch's jacrev and vmap.
        # (Assumes that func_cpu returns an output of the same dimension as its input.)
        jacobian_fn = jacrev(get_flow)
        # vmap over the batch:
        J_all = torch.vmap(jacobian_fn)(sample_x)  # shape: [N, num_variables, num_variables]
        # Average the Jacobians and take the negative (as in train_ngm_sf2m)
        A_estim = -J_all.mean(dim=0).numpy()
        # Extract the causal graph from the hidden layer weights.
        W_v = func.causal_graph(w_threshold=0.0)
        # Compute y_true from A_true as in train_ngm_sf2m:
        y_true = np.abs(np.sign(A_true).astype(int).flatten())

    return A_estim, W_v, y_true


def jacobian_inference(A_estim, y_true):

    y_pred = np.abs(A_estim.flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)

    return prec, rec, thresh, avg_prec


def causal_graph_extraction(W_v, y_true):

    y_pred_mlp = np.abs(W_v.flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred_mlp)
    avg_prec_mlp = average_precision_score(y_true, y_pred_mlp)

    return prec, rec, thresh, avg_prec_mlp


def train_ngm_sf2m(A_true, xs, n, sigma, N=500, T=5, t1=2.5):

    # Replace the MLP flow model with MLPODEF, keep MLP for score
    dims = [n, 64, 64, 1]
    s = fm.MLP(d=n, hidden_sizes=[64, 64], time_varying=True)
    v = MLPODEF(dims=dims, GL_reg=0.01, bias=True, time_invariant=True)
    s = s.to(DEVICE)
    v = v.to(DEVICE)

    # Normalize the input data `xs` for NGM-SF2M
    # xs shape: [num_conditions, T, N_per_cond, n]
    # We normalize across all conditions and time points
    all_cells_list = []
    for cond_idx in range(xs.shape[0]):
        for t_idx in range(xs.shape[1]):
            all_cells_list.append(torch.from_numpy(xs[cond_idx, t_idx]).float())
    
    normalized_data_list, scaler = normalize_data(all_cells_list)
    
    # Reconstruct the normalized xs structure (assuming single condition for now, as per original code)
    # Original code assumes xs[0, ...]
    if xs.shape[0] == 1:
        normalized_xs_grouped = normalized_data_list # List of tensors [T, N, n]
        # Convert list of tensors back to a single tensor for EntropicOTFM input
        # This might need adjustment if EntropicOTFM expects a different structure
        # or if multiple conditions are used later.
        # Stacking normalized tensors for each time point.
        normalized_xs_tensor = torch.stack(normalized_xs_grouped, dim=0).unsqueeze(0) # Shape: [1, T, N, n]
    else:
        # Handle multiple conditions if necessary - requires restructuring normalized_data_list
        raise NotImplementedError("Normalization for multiple conditions in NGM-SF2M needs restructuring")


    optim = torch.optim.AdamW(list(s.parameters()) + list(v.parameters()), 3e-3)

    # Use the normalized data to initialize EntropicOTFM
    model = fm.EntropicOTFM(
        # Pass the reshaped normalized tensor: [T*N, n]
        # Use clone().detach() to avoid warning and ensure a clean copy
        normalized_xs_tensor[0].reshape(-1, n).clone().detach(), 
        torch.tile(torch.arange(T), (N, 1)).T.reshape(-1),
        dt=t1 / T,
        sigma=sigma[0, 0],
        T=T,
        dim=n,
        device=DEVICE, # Pass device explicitly if EntropicOTFM expects it
    )

    trace = []
    alpha = 0.1
    reg = 1e-5
    training_steps = 4_000
    for i in tqdm(range(training_steps)):
        _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(batch_size=64)
        optim.zero_grad()

        # Reshape inputs for MLPODEF
        s_input = _x.unsqueeze(1).to(DEVICE)
        v_input = _x.unsqueeze(1).to(DEVICE)
        t_input = _t.unsqueeze(1).to(DEVICE)

        # Get model outputs and reshape
        s_fit = s(_t, _x).squeeze(1)
        # v_fit = v(t_input, v_input).squeeze(1)
        v_fit = v(t_input, v_input).squeeze(1) - model.sigma**2 / 2 * s(_t, _x)

        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)

        L_reg = v.l2_reg() + v.fc1_reg()
        L = alpha * L_score + (1 - alpha) * L_flow + reg * L_reg

        with torch.no_grad():
            if i % 500 == 0:
                print(L_score.item(), L_flow.item(), L_reg.item())
            trace.append(L.item())

        L.backward()
        optim.step()

        # proximal(s.fc1.weight, s.dims, lam=s.GL_reg, eta=0.01)
        proximal(v.fc1.weight, v.dims, lam=v.GL_reg, eta=0.01) 

        # Send the model to the device
        # Plan is only used to sample the batches
        # EOT plan is not moved to GPU, all done on CPU

    # Plot training loss
    # plt.plot(trace)
    # plt.title("Training Loss")
    # plt.show()

    with torch.no_grad():
        v_cpu = v.cpu()
        def get_flow(t, x):
            x_input = x.unsqueeze(0)  # Add batch dimension
            t_input = t.unsqueeze(0).unsqueeze(0)  # [1, 1, 1]
            return v_cpu(t_input, x_input).squeeze(0).squeeze(0)
        
        Ju = jacrev(get_flow, argnums=1)
        J = []
        for i in range(model.T):
            t = torch.tensor(i * model.dt, device="cpu")
            batch_x = model.x[model.t_idx == 0, :].cpu()
            J_t = torch.vmap(lambda x: Ju(t, x))(batch_x)
            J.append(J_t)

    # Convert to numpy and ensure correct dimensions
    A_estim = -sum([x.mean(0) for x in J]) / len(J)
    A_estim = A_estim.cpu().numpy()
    W_v = v.causal_graph(w_threshold=0.0)
    y_true = np.abs(np.sign(A_true).astype(int).flatten())
    return A_estim, W_v, y_true


def main():
    ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000]
    ngm_sf2m_times = []
    discrete_ode_times = []
    jacob_aupr_ngm = []
    causal_graph_aupr_ngm = []
    jacob_aupr_disc = []
    causal_graph_aupr_disc = []

    for n in ns:
        print(f"\nRunning experiment for n = {n}...")
        xs, A_true, sigma = generate_linear_system(n)
        
        # ---- NGM-NODE Training ----
        hidden_dim = 100
        dims = [n, hidden_dim, 1]
        func = MLPODEF(dims=dims, GL_reg=0.01).to(DEVICE)
        xs_wt = xs[0]
        grouped_data = [torch.from_numpy(xs_wt[t]).float() for t in range(xs_wt.shape[0])]
        normalized_data, scaler = normalize_data(grouped_data)
        
        disc_start_time = time.time()
        A_estim_disc, W_v_disc, y_true_disc = train_random_discrete_time_ode(
            func, normalized_data, A_true, device=DEVICE)
        disc_elapsed = time.time() - disc_start_time
        discrete_ode_times.append(disc_elapsed)
        
        _, _, _, avg_prec_disc_causal = causal_graph_extraction(W_v_disc, y_true_disc)
        causal_graph_aupr_disc.append(avg_prec_disc_causal)
        _, _, _, avg_prec_disc_jacob = jacobian_inference(A_estim_disc, y_true_disc)
        jacob_aupr_disc.append(avg_prec_disc_jacob)


        # ---- NGM-SF2M Training ----
        ngm_sf2m_start_time = time.time()
        A_estim_ngm, W_v_ngm, y_true_ngm = train_ngm_sf2m(A_true, xs, n, sigma)
        ngm_sf2m_elapsed = time.time() - ngm_sf2m_start_time
        ngm_sf2m_times.append(ngm_sf2m_elapsed)
        
        _, _, _, avg_prec_ngm_causal = causal_graph_extraction(W_v_ngm, y_true_ngm)
        causal_graph_aupr_ngm.append(avg_prec_ngm_causal)
        _, _, _, avg_prec_ngm_jacob = jacobian_inference(A_estim_ngm, y_true_ngm)
        jacob_aupr_ngm.append(avg_prec_ngm_jacob)
        

    print("Training complete.")

    results_df = pd.DataFrame({
        "Number of Variables (n)": ns,
        "NGM-SF2M Time (s)": ngm_sf2m_times,
        "NGM-NODE Time (s)": discrete_ode_times,
        "NGM-SF2M Causal Graph AUPR": causal_graph_aupr_ngm,
        "NGM-NODE Causal Graph AUPR": causal_graph_aupr_disc
    })
    
    results_df.to_csv("results.csv", index=False)
    print("Saved numerical results to 'results.csv'.")

    # Plotting: Compare AUPR and training times for NGM-SF2M vs. Discrete-Time ODE.
    fig, ax1 = plt.subplots(figsize=(3, 3))
    ax2 = ax1.twinx()

    # Plot AUPR values
    # ax1.plot(ns, jacob_aupr_ngm, 'o-', color='blue', label='NGM-SF2M Jacobian AUPR')
    # ax1.plot(ns, jacob_aupr_disc, 'o--', color='cyan', label='Discrete ODE Jacobian AUPR')
    ax1.plot(ns, causal_graph_aupr_ngm, 's-', color='salmon', label='NGM-SF2M Causal Graph AUPR')
    ax1.plot(ns, causal_graph_aupr_disc, 's--', color='lightblue', label='NGM-NODE Causal Graph AUPR')
    ax1.set_xlabel("Number of Variables (n)", fontsize=12)
    ax1.set_ylabel("AUPR", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(ns)

    # Plot training times
    ax2.plot(ns, ngm_sf2m_times, '^-', color='lightgreen', linestyle='--', label='NGM-SF2M Time (s)')
    ax2.plot(ns, discrete_ode_times, 'v-', color='blue', linestyle='--', label='NGM-NODE Time (s)')
    ax2.set_ylabel("Time (seconds)", fontsize=12)
    
    # Combine legends from both axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    # # Add direct text labels near the end of each curve
    # texts = []
    # texts.append(ax1.text(ns[-1], causal_graph_aupr_ngm[-1], "NGM-SF2M Causal Graph AUPR", fontsize=10, color='green', va='bottom'))
    # texts.append(ax1.text(ns[-1], causal_graph_aupr_disc[-1], "NGM-NODE Causal Graph AUPR", fontsize=10, color='lime', va='bottom'))

    # texts.append(ax2.text(ns[-1], ngm_sf2m_times[-1], "NGM-SF2M Time (s)", fontsize=10, color='black', va='bottom'))
    # texts.append(ax2.text(ns[-1], discrete_ode_times[-1], "NGM-NODE Time (s)", fontsize=10, color='red', va='bottom'))

    # # Adjust labels to avoid overlaps
    # adjust_text(texts)

    plt.title("AUPR and Training Time vs. Number of Variables", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




