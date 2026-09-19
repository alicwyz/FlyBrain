import numpy as np

def train(ridge_lambda=1e-3):
	"""Fit a linear readout (any number of output rows) from the logged
	reservoir states to the logged targets, entirely in TD -- no CSV, no
	external script. Ridge regression closed form:
		W = Y^T X^T (X X^T + lambda*I)^-1
	X: [T, N_states] logged states. Y: [T, N_out] logged targets.
	W ends up [N_out, N_states] -- N_out is whatever train_target/
	train_targets_log happens to have: 1, 3, 100, anything. The math
	doesn't care, it just changes W's row count.
	"""
	states_log = op('sample_table')
	targets_log = op('target_table')
	T = states_log.numRows
	N_states = states_log.numCols
	N_out = targets_log.numCols
	if T == 0:
		print('No logged rows yet -- let train_recorder run first.')
		return None

	X = np.array([[float(states_log[r, c].val or 0) for c in range(N_states)] for r in range(T)])
	Y = np.array([[float(targets_log[r, c].val or 0) for c in range(N_out)] for r in range(T)])

	Xt = X.T
	reg = ridge_lambda * np.eye(N_states)
	W = Y.T @ Xt.T @ np.linalg.pinv(Xt @ Xt.T + reg)  # [N_out, N_states]

	out = op('readout_weights')
	out.clear()
	for row in W:
		out.appendRow([repr(float(v)) for v in row])
	out.store("W", W)

	pred = X @ W.T
	mse = float(np.mean((pred - Y) ** 2))
	var = float(np.var(Y)) or 1.0
	
	info = op("training_info")
	
	info[1,"targets"] = N_out
	info[1,"samples"] = N_states
	info[1,"size"] = T
	info[1,"MSE"] = mse
	info[1,"normalized"] = mse/var
	
	print(f'Trained readout: {N_out} outputs x {N_states} states, {T} samples. '
	     f'MSE={mse:.6f} normalized={mse/var:.4f} (0=perfect, 1=no better than predicting the mean)')
	return W
	
train()