#ifndef GlobalNodes_H_
#define GlobalNodes_H_

#include "ModelParams.h"

struct GlobalNodes {
	vector<LookupNode> word_inputs;
	vector<LookupNode> ext_word_inputs;
	vector<ConcatNode> word_represents;

	LSTM1Builder word_lstm_left;
	LSTM1Builder word_lstm_right;

	inline void resize(const int &maxsize) {
		word_inputs.resize(maxsize);
		ext_word_inputs.resize(maxsize);
		word_represents.resize(maxsize);
		word_lstm_left.resize(maxsize);
		word_lstm_right.resize(maxsize);
	}

	inline void initial(ModelParams &params, const HyperParams &hyparams){
		resize(max_length);
		int maxsize = word_inputs.size();
		for(int idx = 0; idx < maxsize; idx++) { 
			word_inputs[idx].setParam(&params.wordEmb);
			word_inputs[idx].init(hyparams.wordDim, hyparams.dropProb);
			ext_word_inputs[idx].setParam(&params.extWordEmb);
			ext_word_inputs[idx].init(hyparams.extWordDim, hyparams.dropProb);
			word_represents[idx].init(hyparams.wordRepresentDim, -1);
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
			const string cur_word = inst.words[idx];
			word_inputs[idx].forward(cg, cur_word);
			ext_word_inputs[idx].forward(cg, cur_word);
		}
		for (int idx = 0; idx < word_size; idx++) {
			word_represents[idx].forward(cg, &word_inputs[idx], &ext_word_inputs[idx]);
		}
		word_lstm_left.forward(cg, getPNodes(word_represents, word_size));
		word_lstm_right.forward(cg, getPNodes(word_represents, word_size));

	}
};

#endif /* GlobalNodes_H_ */