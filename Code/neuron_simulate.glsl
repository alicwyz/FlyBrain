void main() {
	uint id = TDIndex();
	if (id >= TDNumElements()) return;

	// Load attributes
	bool isInput = (TDInPoint_is_input() == 1);
	bool isOutput = (TDInPoint_is_output() == 1);
	
	// Store last state
	float prevState = TDInPoint_state();
	
	// Array size for max connections
	int MAX_K = min(uDATA_K, uMAX_K);
	
	// Loop for all connections
	float gather = 0.0;
	for (int k = 0; k < MAX_K; k++) {
		int srcIdx = int(TDInPoint_src(0, id, k));
		if (srcIdx < 0) continue;
		float w = TDInPoint_weight(0, id, k);
		float srcState = TDInPoint_state(0, srcIdx);
		gather += w * srcState;
	}
	
	// Gather input
	float inputTerm = 0.0;
	if (isInput) {
		int slot = int(TDInPoint_local_idx());
		ivec2 texSize = textureSize(uInputTex, 0);
		if (texSize.x > 0) {
			int x = slot % texSize.x;
			float sig = texelFetch(uInputTex, ivec2(x, 0), 0).r;
			inputTerm = sig;
		}
	}
	
	// Weighted input
	float drive = gather * uWeightScale.x + inputTerm;
	// Normalize
	float target = tanh(drive);
	// Leaky filter
	float newState = mix(prevState, target, uLeak.x);
	
	oTDPoint_state[id] = newState;
}
