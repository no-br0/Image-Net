from src.backend_cupy import to_cpu, to_device
import cupy as cp
from Config.log_dir import TELEMETRY_LOG_FOLDER
import os, json

class SpeculativeBatchLion:
    def __init__(self, params={}):
        self.name = params.get("name", "lion")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.99)
        self.weight_decay = params.get("weight_decay", 0.0)
        self.momentum_W = {}
        self.momentum_b = {}
        
        self.prev_weight = {}
        self.prev_bias = {}
        self.telemetry = {}
        self.prev_loss = {}


    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "weight_decay": self.weight_decay,
            "momentum_W": {k: to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {k: to_cpu(v) for k, v in self.momentum_b.items()},
            "prev_weight": {k: to_cpu(v) for k, v in self.prev_weight.items()},
            "prev_bias": {k: to_cpu(v) for k, v in self.prev_bias.items()},
            "prev_loss": {k: to_cpu(v) for k, v in self.prev_loss.items()},
            "prev_loss": {int(k): {int(i): j for i, j in v.items()} for k, v in self.prev_loss.items()},
        }
        
        
    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.momentum_W = {
            int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()
        }
        self.momentum_b = {
            int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()
        }
        self.prev_weight = {
            int(k): to_device(v) for k, v in state.get("prev_weight", {}).items()
        }
        self.prev_bias = {
            int(k): to_device(v) for k, v in state.get("prev_bias", {}).items()
        }
        self.prev_loss = {
        int(k): {int(i): to_device(j) for i, j in v.items()} for k, v in state.get("prev_loss", {}).items()
        }


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
        
        
        total = 0
        reverted = 0
        for batch_idx, layer_dict in self.telemetry.items():
            for layer_idx, did_revert in layer_dict.items():
                total += 1
                reverted += did_revert
        if self.telemetry.__len__() is not 0:
            reverted_epoch = reverted / len(self.telemetry[1].keys())
        else:
            reverted_epoch = 0.0
        revert_percentage = reverted / total if total > 0 else 0.0
        
        entry = {"global_epoch": epoch}
        entry["0_reverted"] = float(reverted_epoch)
        entry["1_revert_percentage"] = revert_percentage
        
        
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error:", e)


    def step(self, model, layer_idx, grad_W, grad_b):
        
        if model.batch_index not in self.telemetry:
            self.telemetry[model.batch_index] = {}
            
        if model.batch_index not in self.prev_loss:
            self.prev_loss[model.batch_index] = {}
        
        if self.prev_weight is None:
            self.prev_weight = {k: cp.zeros_like(v) for k, v in model.weights.items()}
            
        if self.prev_bias is None:
            self.prev_bias = {k: cp.zeros_like(v) for k, v in model.bias.items()}
        
        if model.current_loss is not None and self.prev_loss.__contains__(model.batch_index):
            # is this the proper implementation for reverting the weights and biases
            if self.prev_loss[model.batch_index].__contains__(layer_idx):
                if model.current_loss > self.prev_loss[model.batch_index][layer_idx]:
                    
                    model.weights[layer_idx][...] = self.prev_weight[layer_idx]
                    model.bias[layer_idx][...] = self.prev_bias[layer_idx]
                    self.telemetry[model.batch_index][layer_idx] = 1
                    #print(1)
                else:
                    self.telemetry[model.batch_index][layer_idx]= 0
                    #print(2)
            else:
                self.telemetry[model.batch_index][layer_idx] = 0
        else:
            self.telemetry[model.batch_index][layer_idx] = 0
            #print(3)
        
        self.prev_loss[model.batch_index][layer_idx] = model.current_loss
        
        # Initialize buffers if needed
        if layer_idx not in self.momentum_W:
            self.momentum_W[layer_idx] = cp.zeros_like(grad_W)
            self.momentum_b[layer_idx] = cp.zeros_like(grad_b)

        m_W = self.momentum_W[layer_idx]
        m_b = self.momentum_b[layer_idx]

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        self.prev_weight[layer_idx] = model.weights[layer_idx].copy()
        self.prev_bias[layer_idx] = model.bias[layer_idx].copy()

        model.weights[layer_idx] -= model.learning_rate * cp.sign(m_W)
        model.bias[layer_idx] -= model.learning_rate * cp.sign(m_b)

        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b