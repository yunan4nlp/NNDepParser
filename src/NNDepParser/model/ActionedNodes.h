#ifndef ACTIONED_NODES_H 
#define ACTIONED_NODES_H

#include "ModelParams.h"
#include "AtomFeature.h"

// score the action one by one
class ActionedNodes {
public:

	vector<LookupNode> current_action_input;
	vector<PDotNode> action_score;
	vector<PAddNode> outputs;

	ConcatNode state_concat;
	UniNode state_hidden;
	BucketNode bucket_word;

	const HyperParams *pOpts;

	inline void initial(ModelParams &params, const HyperParams &hyparams) {
		current_action_input.resize(hyparams.actionNum);
		action_score.resize(hyparams.actionNum);
		outputs.resize(hyparams.actionNum);
		for (int idx = 0; idx < hyparams.actionNum; idx++) {
			current_action_input[idx].setParam(&params.actionEmb);
			current_action_input[idx].init(hyparams.actionDim, hyparams.dropProb);
			action_score[idx].init(1, -1);
			outputs[idx].init(1, -1);
		}
		bucket_word.init(hyparams.rnnHiddenSize, -1);
		state_concat.init(hyparams.rnnHiddenSize * 2, -1);
		state_hidden.setParam(&params.state_hidden_params);
		state_hidden.init(hyparams.actionDim, hyparams.dropProb);
		pOpts = &hyparams;
	}

	inline void forward(Graph *cg, const vector<CAction> &actions, const AtomFeat &atomFeat) {
		int action_num = actions.size();
		CAction ac;
		bucket_word.forward(cg, 0);
		const PNode pword_lstm_left = 
			atomFeat._next_index >= 0 ? (const PNode)&atomFeat._pword_lstm_left->_hiddens[atomFeat._next_index] : (const PNode)&bucket_word;
		const PNode pword_lstm_right = 
			atomFeat._next_index >= 0 ? (const PNode)&atomFeat._pword_lstm_right->_hiddens[atomFeat._next_index] : (const PNode)&bucket_word;
		vector<PNode> feats;
		feats.push_back(pword_lstm_left);
		feats.push_back(pword_lstm_right);
		state_concat.forward(cg, feats);
		state_hidden.forward(cg, &state_concat);
		vector<PNode> sumNodes;
		for (int idx = 0; idx < action_num; idx++) {
			sumNodes.clear();
			ac.set(actions[idx]);
			const string &action = ac.str(*pOpts);
			current_action_input[idx].forward(cg, action);
			action_score[idx].forward(cg, &current_action_input[idx], &state_hidden);
			sumNodes.push_back(&action_score[idx]);
			outputs[idx].forward(cg, sumNodes);
		}
	}
	
};

#endif /*ACTIONED_NODES_H*/