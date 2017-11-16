#ifndef GlobalNodes_H_
#define GlobalNodes_H_

#include "ModelParams.h"

struct GlobalNodes {
	vector<LookupNode> word_inputs;
	LSTM1Builder word_lstm_left;
	LSTM1Builder word_lstm_right;

	inline void resize(const int &maxsize) {
		word_inputs.resize(maxsize);
		word_lstm_left.resize(maxsize);
		word_lstm_right.resize(maxsize);
	}

	inline void initial(ModelParams &params, const HyperParams &hyparams){
		resize(max_length);
		int maxsize = word_inputs.size();
		for(int idx = 0; idx < maxsize; idx++) { 
			word_inputs[idx].setParam(&params.wordEmb);
			word_inputs[idx].init(hyparams.wordDim, hyparams.dropProb);
		}
		word_lstm_left.init(&params.word_lstm_left_params, hyparams.dropProb, true);
		word_lstm_right.init(&params.word_lstm_right_params, hyparams.dropProb, false);
	}

	inline void forward(Graph *cg, const Instance &inst) {
		int word_size = inst.words.size();
		int max_size = word_inputs.size();
		if (word_size > max_size)
			word_size = max_size;
		for (int idx = 0; idx < word_size; idx++) { 
			word_inputs[idx].forward(cg, inst.words[idx]);
		}
		word_lstm_left.forward(cg, getPNodes(word_inputs, word_size));
		word_lstm_right.forward(cg, getPNodes(word_inputs, word_size));

	}
};

#endif /* GlobalNodes_H_ */