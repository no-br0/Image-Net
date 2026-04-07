# worker_train.py
import cupy as cp
from src.train import train_streaming
from src.neural_net import NeuralNet
from src.loss_registry import LOSS_REGISTRY

def worker_main(conn, model_state, epochs, batch_size, loss_name, shuffle):
    model = NeuralNet.from_state(model_state)

    for _ in range(epochs):
        # run exactly ONE epoch
        timing_log = train_streaming(
            model,
            epochs=1,
            batch_size=batch_size,
            shuffle=shuffle,
            error_func=LOSS_REGISTRY[loss_name],
            telemetry_logger=None,
        )

        # send updated model to main
        conn.send(("epoch", {
			"state": model.to_state(),
			"timing": timing_log,	
		}))

        # wait for main to tell us to continue
        cmd = conn.recv()
        if cmd != "continue":
            break

    # final state for this chunk
    conn.send(("done", model.to_state()))
    conn.close()

    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
