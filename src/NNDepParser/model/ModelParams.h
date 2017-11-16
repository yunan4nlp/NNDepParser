#ifndef ModelParams_H_
#define ModelParams_H_

#include "HyperParams.h"

class ModelParams {
public:
	LookupTable wordEmb;
	LookupTable extWordEmb;
	LSTM1Params word_lstm_left_params;
	LSTM1Params word_lstm_right_params;

	LookupTable actionEmb;
	UniParams state_hidden_params;
public:
	inline bool initial(const HyperParams &opts){
		word_lstm_left_params.initial(opts.rnnHiddenSize, opts.wordRepresentDim);
		word_lstm_right_params.initial(opts.rnnHiddenSize, opts.wordRepresentDim);
		state_hidden_params.initial(opts.actionDim, opts.rnnHiddenSize * 2, true);
		return true;
	}

	inline void exportModelParams(ModelUpdate &ada){
		wordEmb.exportAdaParams(ada);
		extWordEmb.exportAdaParams(ada);
		word_lstm_left_params.exportAdaParams(ada);
		word_lstm_right_params.exportAdaParams(ada);
		actionEmb.exportAdaParams(ada);
		state_hidden_params.exportAdaParams(ada);
	}

	void saveModel() {}

	void loadModel() {}

};

#endif /* ModelParams_H_ */