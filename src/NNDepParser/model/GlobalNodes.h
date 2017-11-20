#ifndef GlobalNodes_H_
#define GlobalNodes_H_

#include "ModelParams.h"

struct GlobalNodes {
	vector<LookupNode> word_inputs;
	vector<LookupNode> ext_word_inputs;
	vector<TriNode> word_represents;

	LSTM1Builder word_lstm_left_layer1;
	LSTM1Builder word_lstm_right_layer1;

	vector<BiNode> word_lstm_combine1;

	LSTM1Builder word_lstm_left_layer2;
	LSTM1Builder word_lstm_right_layer2;

	vector<ConcatNode> word_lstm_concat2;

	vector<LookupNode> tag_inputs;

	inline void resize(const int &maxsize) {
		word_inputs.resize(maxsize);
		ext_word_inputs.resize(maxsize);
		word_represents.resize(maxsize);
		word_lstm_left_layer1.resize(maxsize);
		word_lstm_right_layer1.resize(maxsize);
		word_lstm_combine1.resize(maxsize);
		word_lstm_left_layer2.resize(maxsize);
		word_lstm_right_layer2.resize(maxsize);
		word_lstm_concat2.resize(maxsize);
		tag_inputs.resize(maxsize);
	}

	inline void initial(ModelParams &params, const HyperParams &hyparams){
		resize(max_length);
		int maxsize = word_inputs.size();
		for(int idx = 0; idx < maxsize; idx++) { 
			word_inputs[idx].setParam(&params.word_table);
			word_inputs[idx].init(hyparams.wordDim, hyparams.dropProb);
			ext_word_inputs[idx].setParam(&params.ext_word_table);
			ext_word_inputs[idx].init(hyparams.extWordDim, hyparams.dropProb);
			word_represents[idx].setParam(&params.word_combine_params);
			word_represents[idx].init(hyparams.wordRepresentHiddenSize, hyparams.dropProb);

			tag_inputs[idx].setParam(&params.tag_table);
			tag_inputs[idx].init(hyparams.tagDim, hyparams.dropProb);
		}
		word_lstm_left_layer1.init(&params.word_lstm_left_layer1_params, hyparams.dropProb, true);
		word_lstm_right_layer1.init(&params.word_lstm_right_layer1_params, hyparams.dropProb, false);
		
		for (int idx = 0; idx < maxsize; idx++) {
			word_lstm_combine1[idx].setParam(&params.word_lstm_combine1_params);
			word_lstm_combine1[idx].init(hyparams.hiddenSize,  hyparams.dropProb);
		}
		word_lstm_left_layer2.init(&params.word_lstm_left_layer2_params, hyparams.dropProb, true);
		word_lstm_right_layer2.init(&params.word_lstm_right_layer2_params, hyparams.dropProb, false);

		for (int idx = 0; idx < maxsize; idx++) {
			word_lstm_concat2[idx].init(hyparams.rnnHiddenSize * 4, -1);
		}
	}

	inline void forward(Graph *cg, const Instance &inst) {
		int word_size = inst.words.size();
		int max_size = word_inputs.size();
		if (word_size > max_size)
			word_size = max_size;
		for (int idx = 0; idx < word_size; idx++) { 
			const string &cur_word = inst.words[idx];
			word_inputs[idx].forward(cg, cur_word);
			ext_word_inputs[idx].forward(cg, cur_word);
			const string &cur_tag = inst.tags[idx];
			tag_inputs[idx].forward(cg, cur_tag);
		}

		for (int idx = 0; idx < word_size; idx++) {
			word_represents[idx].forward(cg, &word_inputs[idx], &ext_word_inputs[idx], &tag_inputs[idx]);
		}
		word_lstm_left_layer1.forward(cg, getPNodes(word_represents, word_size));
		word_lstm_right_layer1.forward(cg, getPNodes(word_represents, word_size));

		for (int idx = 0; idx < word_size; idx++) {
			word_lstm_combine1[idx].forward(cg, &word_lstm_left_layer1._hiddens[idx], &word_lstm_right_layer1._hiddens[idx]);
		}
		word_lstm_left_layer2.forward(cg, getPNodes(word_lstm_combine1, word_size));
		word_lstm_right_layer2.forward(cg, getPNodes(word_lstm_combine1, word_size));
		for (int idx = 0; idx < word_size; idx++) {
			word_lstm_concat2[idx].forward(cg, 
				&word_lstm_left_layer1._hiddens[idx], &word_lstm_right_layer1._hiddens[idx],
				&word_lstm_left_layer2._hiddens[idx], &word_lstm_right_layer2._hiddens[idx]);
		}
	}
};

#endif /* GlobalNodes_H_ */