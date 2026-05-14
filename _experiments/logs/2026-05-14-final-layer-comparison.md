# Final Layer Width Comparison (32 vs 64 vs 384)

### Config

- Patterns: [2026-05-14](../configs/patterns/2026-05-14.md)
- Other: [2026-05-14](../configs/other/2026-05-14.md)
- Num of training images: 4

### Topologies Tested

- __Variant A — 32-wide final layer__<br>
`HIDDEN_LAYER_TOPOLOGY = [576, 512, 448, 32]`

- __Variant B — 64-wide final layer__<br>
`HIDDEN_LAYER_TOPOLOGY = [576, 512, 448, 64]`

- __Variant C — 384-wide final layer__<br>
`HIDDEN_LAYER_TOPOLOGY = [576, 512, 448, 384]`

### Observations

#### 1. Oscillation Behaviour (Expected for Pairwise Training)
- Training uses all __6 image pairs__ formed from the 4 training images.<br>
Each epoch trains on __one pair__ (~4M samples, 106 batches).<br>
The order of the 6 pairs is reshuffled each lap using a fixed seed, so the model sees the same set of pairs but in __varying orders__ across cycles.<br>
Switching to a new pair each epoch produces the visible __up/down jumps__ in the loss.
- Pairwise training produces __large loss oscillations__, especially early in training.<br>
These oscillations are __not__ ±2800 swings — the loss moves __within a band__ typically ranging from ~2800 (bottom) to ~3400 (top).
- The __oscillation band__ refers to this vertical range (top minus bottom).
- Large oscillation bands are __normal__ for this training regime and do __not__ indicate instability.
- Despite the amplitude, the __shape and timing__ of the oscillations are extremely similar across all three variants.
- Across all variants, the oscillation band __gradually narrows__ over time.<br>
The top and bottom of the band __move closer together__ as training progresses.

#### 2. Early-Stage Dynamics (Epochs 0-1000)
- The __64-wide__ final layer has the __narrowest band__ during the first ~1000 epochs.
- The __32-wide__ final layer also has a __narrower band__ than 384, but __not__ as narrow as 64.
- Both 64 and 32 reach __lower loss values earlier__ than 384, with 64 descending the fastest.
- while 32 and 64 share the same overall oscillation timing and structure, they are __not__ equally strong:
	- 64 oscillates __tighter__
	- 64 descends __faster__
	- 64 reaches __deeper earlier loss basins__
	- 32 behaves like a __weaker, slower__ version of the same trajectory
- The __384-wide__ variant shows the __widest band__ and the __slowest early descent__.

#### 3. Mid-Training Dynamics (Epochs 1000-1600)
(32-wide __not__ yet tested to this stage)
- Between ~1000 and 1600 epochs, the __64-wide__ and __384-wide__ variants show __similar__ band widths.
- Their loss values also become __similar__ in this region, indicating a partial convergence of behaviour.