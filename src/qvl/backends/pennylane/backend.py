"""PennyLane backend implementation with 2-qubit VQC."""

import time
import numpy as np
from typing import Dict, Any

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    raise ImportError(
        "PennyLane is required for this backend. "
        "Install with: pip install -e '.[pennylane]'"
    )


class PennyLaneBackend:
    """2-qubit variational quantum circuit backend."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize PennyLane backend.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.seed = config.get('seed', 0)
        self.training_config = config.get('training', {})
        self.noise_config = config.get('noise', {})

        self.n_samples = self.training_config.get('n_samples', 100)
        self.n_qubits = 2
        self.n_layers = self.training_config.get('n_layers', 2)
        self.n_epochs = self.training_config.get('n_epochs', 100)
        self.learning_rate = self.training_config.get('learning_rate', 0.1)

        self.depolarizing_p = self.noise_config.get('depolarizing_p', 0.0)
        self.measurement_bitflip_p = self.noise_config.get('measurement_bitflip_p', 0.0)
        self.amplitude_gamma = self.noise_config.get('amplitude_gamma', 0.0)

        np.random.seed(self.seed)

        self.dev = qml.device('default.mixed', wires=self.n_qubits)

        # Create QNode with closure to ensure proper binding
        def circuit_fn(x, params):
            """Full quantum circuit."""
            self._feature_map(x)
            self._ansatz(params)
            return qml.expval(qml.PauliZ(0))

        self._qnode = qml.QNode(circuit_fn, self.dev, interface='autograd')

    def run(self) -> Dict[str, Any]:
        """Run VQC experiment.

        Returns:
            Results dictionary
        """
        start_time = time.time()

        X, y = self._generate_data()

        params = self._initialize_params()

        params, final_loss = self._train(X, y, params)

        predictions = self._predict(X, params)

        metrics = self._compute_metrics(params, predictions, y, final_loss, X)

        wall_time = time.time() - start_time

        return {
            'metrics': metrics,
            'timing': {
                'wall_time_sec': wall_time,
            },
            'noise': {
                'depolarizing_p': self.depolarizing_p,
                'measurement_bitflip_p': self.measurement_bitflip_p,
                'amplitude_gamma': self.amplitude_gamma,
            },
        }

    def _generate_data(self) -> tuple:
        """Generate synthetic binary classification dataset."""
        n_per_class = self.n_samples // 2

        X_class0 = np.random.randn(n_per_class, self.n_qubits)
        X_class1 = np.random.randn(n_per_class, self.n_qubits) + 1.5

        X = np.vstack([X_class0, X_class1])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def _initialize_params(self):
        """Initialize variational parameters."""
        n_params = self.n_qubits * self.n_layers * 3
        params = np.random.randn(n_params) * 0.1
        return pnp.array(params, requires_grad=True)

    def _feature_map(self, x):
        """Simple angle encoding feature map."""
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)

    def _ansatz(self, params):
        """Variational ansatz with rotation and entangling layers."""
        idx = 0
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.RX(params[idx], wires=qubit)
                idx += 1
                qml.RY(params[idx], wires=qubit)
                idx += 1
                qml.RZ(params[idx], wires=qubit)
                idx += 1

            if self.depolarizing_p > 0:
                for qubit in range(self.n_qubits):
                    qml.DepolarizingChannel(self.depolarizing_p, wires=qubit)

            if layer < self.n_layers - 1:
                qml.CNOT(wires=[0, 1])

                if self.depolarizing_p > 0:
                    for qubit in range(self.n_qubits):
                        qml.DepolarizingChannel(self.depolarizing_p / 2, wires=qubit)

        if self.amplitude_gamma > 0:
            for qubit in range(self.n_qubits):
                qml.AmplitudeDamping(self.amplitude_gamma, wires=qubit)


    def _predict_single(self, x, params):
        """Predict for single sample."""
        expval = self._qnode(x, params)

        prob = (expval + 1) / 2

        if self.measurement_bitflip_p > 0:
            if np.random.rand() < self.measurement_bitflip_p:
                prob = 1 - prob

        return prob

    def _predict(self, X, params):
        """Make predictions for all samples."""
        return pnp.array([self._predict_single(x, params) for x in X])

    def _loss(self, params, X, y):
        """Binary cross-entropy loss."""
        predictions = self._predict(X, params)
        predictions = pnp.clip(predictions, 1e-10, 1 - 1e-10)
        loss = -pnp.mean(y * pnp.log(predictions) + (1 - y) * pnp.log(1 - predictions))
        return loss

    def _train(self, X, y, params):
        """Train the VQC using gradient descent."""
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)

        def cost_fn(p):
            return self._loss(p, X, y)

        for epoch in range(self.n_epochs):
            params = opt.step(cost_fn, params)

            if epoch % 20 == 0:
                current_loss = cost_fn(params)

        final_loss = float(cost_fn(params))
        return params, final_loss

    def _compute_metrics(self, params, predictions, targets, loss, X):
        """Compute verification metrics."""
        accuracy = float(np.mean((predictions > 0.5) == targets))

        hessian_min, hessian_max, ident_proxy = self._compute_hessian_proxy(params, X, targets)

        fisher_cond = max(1.0, hessian_max / (hessian_min + 1e-10))
        fisher_rank = float(len(params)) if hessian_min > 1e-8 else 1.0

        return {
            'accuracy': accuracy,
            'loss': loss,
            'ident_proxy': ident_proxy,
            'hessian_min_abs': hessian_min,
            'hessian_max_abs': hessian_max,
            'fisher_condition_number': fisher_cond,
            'fisher_effective_rank': fisher_rank,
            'seed_robustness': 0.0,
        }

    def _compute_hessian_proxy(self, params, X, y):
        """Compute Hessian proxy using finite differences."""
        eps = 1e-4
        n_params = len(params)

        def loss_fn(p):
            return self._loss(p, X, y)

        base_loss = loss_fn(params)

        grad = np.zeros(n_params)
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (loss_fn(params_plus) - base_loss) / eps

        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            return 0.0, 0.0, 0.0

        hessian_diag = np.zeros(n_params)
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            grad_plus = (loss_fn(params_plus) - base_loss) / eps
            grad_minus = (base_loss - loss_fn(params_minus)) / eps
            hessian_diag[i] = abs((grad_plus - grad_minus) / eps)

        hessian_diag = hessian_diag[hessian_diag > 1e-10]

        if len(hessian_diag) == 0:
            return 0.0, 0.0, 0.0

        hessian_min = float(np.min(hessian_diag))
        hessian_max = float(np.max(hessian_diag))
        ident_proxy = float(hessian_min / (hessian_max + 1e-10))

        return hessian_min, hessian_max, ident_proxy
