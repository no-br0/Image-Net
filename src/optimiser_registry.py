from src.optimisers import *


OPTIMISER_REGISTRY = {
    "sgd": SGD,
    "nesterov": Nesterov,
    "lion": Lion,
    "lion_loss_delta": LionLossDelta,
    "lion_oscillation_lr": LionOscillationLR,
    "lion_directional_freeze": LionDirectionalFreeze,
    "lion_cosine_gate": LionCosineGate,
    "lion_refinement_mode": LionRefinementMode,
    "lion_refine_cosine": LionRefineCosine,
    "lion_delta_refine_cosine": LionDeltaRefineCosine,
    "lion_refine_freeze_cosine": LionRefineFreezeCosine,
    "lion_delta_refine_freeze_cosine": LionDeltaRefineFreezeCosine,
    "adabelief": AdaBelief,
    "lion_belief_refine": LionBeliefRefine,
    "qhlion_refine": QHLionRefine,
    "qhlion_belief_refine": QHLionAdaBeliefRefine,
    "stratagum": Stratagum,
}