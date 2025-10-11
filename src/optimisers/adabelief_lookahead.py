import cupy as cp
import os, json
from Config.log_dir import TELEMETRY_LOG_FOLDER

class AdaBeliefLookahead:
    def __init__(self, params):
        # Extract hyperparameters from unified dict
        self.params = params["params"]
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.999)
        self.eps = params.get("eps", 1e-8)
        self.weight_decay = params.get("weight_decay", 0.0)
        self.k = params.get("lookahead_k", 5)
        self.alpha = params.get("lookahead_alpha", 0.5)

        # AdaBelief state
        self.m = [cp.zeros_like(p) for p in self.params]
        self.s = [cp.zeros_like(p) for p in self.params]
        self.t = 0

        # Lookahead state
        self.slow_params = [p.copy() for p in self.params]
        self.step_counter = 0
        
        self.telemetry = {
            "lr": {},
        }

    def get_state(self):
        return {
            "t": self.t,
            "step_counter": self.step_counter,
            "m": [cp.asnumpy(m) for m in self.m],
            "s": [cp.asnumpy(s) for s in self.s],
            "slow_params": [cp.asnumpy(sp) for sp in self.slow_params],
        }

    def load_state(self, state):
        self.t = state["t"]
        self.step_counter = state["step_counter"]
        self.m = [cp.asarray(m) for m in state["m"]]
        self.s = [cp.asarray(s) for s in state["s"]]
        self.slow_params = [cp.asarray(sp) for sp in state["slow_params"]]


    def log_epoch_telemetry(self, epoch):
        try:
            with open(os.path.join("outputs", "current_model_name.json"), "r") as f:
                obj = json.load(f)
                model_name = obj.get("model_name", "nn_model")
        except Exception as e:
            print("Telemetry logging error: could not resolve model name.", e)
            return

        log_path = os.path.join(TELEMETRY_LOG_FOLDER, f"{model_name}_optimiser.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        def mean(vals):
            return float(cp.mean(cp.array(list(vals), dtype=cp.float32))) if vals else 0.0

        entry = {"global_epoch": epoch}

        # Dynamically log only available metrics
        for key, val_dict in self.telemetry.items():
            try:
                entry[key] = mean(val_dict.values())
            except Exception:
                entry[key] = None  # Graceful fallback if values aren't numeric

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error: could not write to log file.", e)



    def step(self, model, layer_idx, grad_W, grad_b):
        self.t += 1
        self.step_counter += 1

        grads = [grad_W, grad_b]
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay != 0:
                g += self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            grad_diff = g - self.m[i]
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (grad_diff ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            s_hat = self.s[i] / (1 - self.beta2 ** self.t)

            update = model.learning_rate * m_hat / (cp.sqrt(s_hat) + self.eps)
            self.params[i] -= update

            # 🔍 Diagnostic hook (optional)
            # model.trace_update(layer_idx, param_idx=i, update=update, grad=g, m_hat=m_hat, s_hat=s_hat)

        if self.step_counter % self.k == 0:
            for i in range(len(self.params)):
                self.slow_params[i] += self.alpha * (self.params[i] - self.slow_params[i])
                self.params[i] = self.slow_params[i].copy()

                # 🔍 Lookahead trace hook (optional)
                # model.trace_lookahead(layer_idx, param_idx=i, fast=self.params[i], slow=self.slow_params[i])
