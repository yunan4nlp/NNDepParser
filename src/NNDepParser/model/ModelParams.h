#ifndef ModelParams_H_
#define ModelParams_H_

#include "HyperParams.h"

class ModelParams {
public:
	LookupTable word_table;
	LookupTable ext_word_table;
	TriParams word_combine_params;
	LSTM1Params word_lstm_left_layer1_params;
	LSTM1Params word_lstm_right_layer1_params;
	BiParams word_lstm_combine1_params;
	LSTM1Params word_lstm_left_layer2_params;
	LSTM1Params word_lstm_right_layer2_params;

	LookupTable tag_table;

	LookupTable scored_action_table;
	UniParams state_hidden_params;
	LookupTable action_table;
	LSTM1Params action_lstm_params;
public:
	inline bool initial(const HyperParams &opts){
		word_combine_params.initial(opts.wordRepresentHiddenSize, opts.wordDim, opts.extWordDim, opts.tagDim, true);
		word_lstm_left_layer1_params.initial(opts.rnnHiddenSize, opts.wordRepresentHiddenSize);
		word_lstm_right_layer1_params.initial(opts.rnnHiddenSize, opts.wordRepresentHiddenSize);
		word_lstm_combine1_params.initial(opts.hiddenSize, opts.rnnHiddenSize, opts.rnnHiddenSize, true);
		word_lstm_left_layer2_params.initial(opts.rnnHiddenSize, opts.hiddenSize);
		word_lstm_right_layer2_params.initial(opts.rnnHiddenSize, opts.hiddenSize);
		state_hidden_params.initial(opts.stateHiddenSize, opts.stateConcatSize, true);
		action_lstm_params.initial(opts.actionHiddenSize, opts.actionDim * 2);
		random_device rd;
		mt19937 gen(rd());
		gen.seed(0);
		std::normal_distribution<> d(0, 1);
		for (int idx = 0; idx < word_table.E.val.size; idx++)
			word_table.E.val.v[idx] = d(gen);
		for (int idx = 0; idx < tag_table.E.val.size; idx++)
			tag_table.E.val.v[idx] = d(gen);
		return true;
	}

	inline void exportModelParams(ModelUpdate &ada){
		word_table.exportAdaParams(ada);
		ext_word_table.exportAdaParams(ada);
		word_combine_params.exportAdaParams(ada);
		word_lstm_left_layer1_params.exportAdaParams(ada);
		word_lstm_right_layer1_params.exportAdaParams(ada);
		word_lstm_combine1_params.exportAdaParams(ada);
		word_lstm_left_layer2_params.exportAdaParams(ada);
		word_lstm_right_layer2_params.exportAdaParams(ada);
		tag_table.exportAdaParams(ada);
		scored_action_table.exportAdaParams(ada);
		state_hidden_params.exportAdaParams(ada);
		action_table.exportAdaParams(ada);
		action_lstm_params.exportAdaParams(ada);
	}

	void saveModel() {}

	void loadModel() {}

};

#endif /* ModelParams_H_ */